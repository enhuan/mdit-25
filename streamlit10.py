import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# --- Load Data (June–Aug) ---
@st.cache_data
def load_data():
    cols = ["date", "item_code", "premise_code", "price"]  # only necessary columns
    june = "https://drive.google.com/uc?id=14g1GiSXFdg_tyWtRlXJbceEyWl3kvgmf"
    july = "https://drive.google.com/uc?id=17AX1AvTTD6-ri8ciZHaMrkSIcec7-Owz"
    aug = "https://drive.google.com/uc?id=1N5wouJhhg_eF-6otEEoLxSTiJJz7jVaT"
    df_list = [
        pd.read_csv(june, usecols=cols, low_memory=True, dtype={"price": "float32"}),
        pd.read_csv(july, usecols=cols, low_memory=True, dtype={"price": "float32"}),
        pd.read_csv(aug, usecols=cols, low_memory=True, dtype={"price": "float32"})
    ]
    df = pd.concat(df_list, ignore_index=True)

    # lookup tables
    item_df = pd.read_csv("https://drive.google.com/uc?id=1pOxjGJOaWKDpQFWR_iKh7prSliQ6f_1X").drop_duplicates(
        "item_code")
    premise_df = pd.read_csv("https://drive.google.com/uc?id=18c_Y_3JXF_G1S3LkUwscgNM743dfhO4f").drop_duplicates(
        "premise_code")
    premise_df = premise_df[["premise_code", "premise", "state", "district"]]

    df = df.merge(item_df, on="item_code", how="left").merge(premise_df, on="premise_code", how="left")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "price"])
    return df


df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")
state_list = df["state"].dropna().unique()
state_choice = st.sidebar.selectbox("Select State", state_list, index=list(state_list).index("Selangor"))
district_choice = st.sidebar.selectbox("Select District", df[df["state"] == state_choice]["district"].dropna().unique())
item_group_choice = st.sidebar.selectbox("Select Item Group", df["item_group"].dropna().unique())
item_category_choice = st.sidebar.selectbox(
    "Select Item Category", df[df["item_group"] == item_group_choice]["item_category"].dropna().unique())
item_choice = st.sidebar.selectbox(
    "Select Item", df[df["item_category"] == item_category_choice]["item"].dropna().unique())
premise_options = df[(df["state"] == state_choice) & (df["district"] == district_choice) & (df["item"] == item_choice)][
    "premise"].dropna().unique()
premise_choice = st.sidebar.selectbox("Select Premise", premise_options)

forecast_date = st.sidebar.date_input("Select Date to Forecast", min_value=pd.to_datetime("2025-09-01"),
                                      max_value=pd.to_datetime("2030-12-31"))
if forecast_date.year == 2025 and forecast_date.month < 9:
    st.warning("Please select a date from September 2025 onwards.")

run_button = st.sidebar.button("Run Forecast")
st.title("Malaysia Item Price Forecasting")


# --- Prophet Model Cache ---
# District-level model: cached for reuse
@st.cache_resource
def fit_prophet_district(df_prophet):
    m = Prophet()
    m.fit(df_prophet)
    return m

# Premise-level model: no cache (always fresh)
def fit_prophet_premise(df_prophet):
    m = Prophet()
    m.fit(df_prophet)
    return m

if run_button:
    filtered = df[
        (df["state"] == state_choice) & (df["district"] == district_choice) & (df["item"] == item_choice)].copy()

    if filtered.empty:
        st.warning("No data found for this selection.")
    else:
        daily = filtered.groupby("date", as_index=False)["price"].mean()
        train_df = daily[(daily["date"] >= "2025-06-01") & (daily["date"] <= "2025-08-31")].copy()

        if len(train_df) < 10:
            st.warning("Not enough training data (June–Aug).")
        else:
            df_prophet = train_df.rename(columns={"date": "ds", "price": "y"})
            model = fit_prophet_district(df_prophet)

            last_train = df_prophet["ds"].max().date()
            sel_date = pd.to_datetime(forecast_date).date()
            periods = (sel_date - last_train).days
            if periods < 1:
                st.warning(f"No future periods to forecast. Training ends on {last_train}.")
            else:
                future = model.make_future_dataframe(periods=periods, freq="D")
                forecast_df = model.predict(future)
                forecast_slice = forecast_df[(forecast_df["ds"] >= pd.to_datetime("2025-09-01")) & (
                            forecast_df["ds"] <= pd.to_datetime(sel_date))][["ds", "yhat", "yhat_lower", "yhat_upper"]]

                # --- Plotting (sample points for efficiency) ---
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(daily["date"].iloc[::3], daily["price"].iloc[::3], label="Historical (Jun–Aug) – District Avg",
                        marker="o", markersize=3)
                ax.plot(forecast_slice["ds"], forecast_slice["yhat"], "r-x", label="Forecast – District Avg")

                # Premise forecast
                if premise_choice:
                    premise_data = filtered[filtered["premise"] == premise_choice].groupby("date", as_index=False)[
                        "price"].mean()
                    ax.plot(premise_data["date"].iloc[::3], premise_data["price"].iloc[::3],
                            label=f"Historical – {premise_choice}", marker="s", markersize=3)

                    train_premise = premise_data[
                        (premise_data["date"] >= "2025-06-01") & (premise_data["date"] <= "2025-08-31")]
                    if len(train_premise) >= 10:
                        df_prophet_premise = train_premise.rename(columns={"date": "ds", "price": "y"})
                        model_premise = fit_prophet_premise(df_prophet_premise)
                        periods_premise = (sel_date - train_premise["date"].max().date()).days
                        if periods_premise >= 1:
                            future_premise = model_premise.make_future_dataframe(periods=periods_premise, freq="D")
                            forecast_premise = model_premise.predict(future_premise)
                            forecast_slice_premise = forecast_premise[
                                (forecast_premise["ds"] >= pd.to_datetime("2025-09-01")) & (
                                            forecast_premise["ds"] <= pd.to_datetime(sel_date))]
                            ax.plot(forecast_slice_premise["ds"], forecast_slice_premise["yhat"], "g--",
                                    label=f"Forecast – {premise_choice}")

                ax.set_title(f"Prophet Forecast for {item_choice} in {district_choice}, {state_choice}")
                ax.set_ylabel("Average Price (RM)")
                ax.legend()
                st.pyplot(fig)

                # --- Forecast Message Boxes ---

                # District forecast for selected date
                district_value = forecast_slice.loc[forecast_slice["ds"] == pd.to_datetime(sel_date), "yhat"]
                if not district_value.empty:
                    st.markdown(
                        f"""
                        <div style='border:1px solid #ddd; padding:10px; border-radius:8px; background-color:#f9f9f9;'>
                            <span style='font-size:14px;'>
                            Prophet's Forecast of Average Price of <b>{item_choice}</b> 
                            in <b>{district_choice}, {state_choice}</b> on <b>{sel_date}</b>: 
                            RM <b>{district_value.values[0]:.2f}</b>
                            </span>
                        </div>
                        """, unsafe_allow_html=True
                    )

                # Premise forecast for selected date
                if premise_choice and len(train_premise) >= 10:
                    premise_value = forecast_slice_premise.loc[
                        forecast_slice_premise["ds"] == pd.to_datetime(sel_date), "yhat"]
                    if not premise_value.empty:
                        pct_diff = ((premise_value.values[0] - district_value.values[0]) / district_value.values[
                            0]) * 100
                        diff_text = f" ({abs(pct_diff):.2f}% {'higher' if pct_diff > 0 else 'lower'} than district average)"
                        st.markdown(
                            f"""
                            <div style='border:1px solid #ddd; padding:10px; border-radius:8px; background-color:#eef9ff;'>
                                <span style='font-size:14px;'>
                                Prophet's Forecast for <b>{item_choice}</b> at <b>{premise_choice}</b> on <b>{sel_date}</b>:  
                                RM <b>{premise_value.values[0]:.2f}</b> {diff_text}
                                </span>
                            </div>
                            """, unsafe_allow_html=True
                        )

                # --- Backtesting & Forecast Tables ---

                def calculate_backtest(y_true, y_pred):
                    mae = mean_absolute_error(y_true, y_pred)
                    mape = (np.abs(y_true - y_pred) / y_true).mean() * 100
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    return mae, mape, rmse


                # District backtest
                y_true_d = df_prophet["y"].values
                y_pred_d = model.predict(df_prophet[["ds"]])["yhat"].values
                mae_d, mape_d, rmse_d = calculate_backtest(y_true_d, y_pred_d)

                # Premise backtest (if enough data)
                if premise_choice and len(train_premise) >= 10:
                    y_true_p = df_prophet_premise["y"].values
                    y_pred_p = model_premise.predict(df_prophet_premise[["ds"]])["yhat"].values
                    mae_p, mape_p, rmse_p = calculate_backtest(y_true_p, y_pred_p)
                else:
                    mae_p = mape_p = rmse_p = None

                # --- Forecast Tables ---

                # District forecast table: only keep Date + District Forecast
                district_table = forecast_slice.rename(
                    columns={"ds": "Date", "yhat": "District Forecast"}
                )[["Date", "District Forecast"]]

                if premise_choice and len(train_premise) >= 10:
                    # Premise forecast table: only keep Date + Premise Forecast
                    premise_table = forecast_slice_premise.rename(
                        columns={"ds": "Date", "yhat": f"{premise_choice} Forecast"}
                    )[["Date", f"{premise_choice} Forecast"]]

                    # Merge with district forecast
                    combined_table = district_table.merge(premise_table, on="Date", how="left")

                    # Add % change column
                    combined_table["Change (%)"] = (
                            (combined_table[f"{premise_choice} Forecast"] - combined_table["District Forecast"]) /
                            combined_table["District Forecast"] * 100
                    )
                else:
                    combined_table = district_table

                st.subheader(f"Forecasted Prices: 2025-09-01 → {sel_date}")
                st.dataframe(combined_table.round(2))

                # --- Display Backtest Accuracy ---
                accuracy_table = pd.DataFrame({
                    "Model": ["District Avg", premise_choice if premise_choice else "N/A"],
                    "MAE": [mae_d, mae_p],
                    "MAPE (%)": [mape_d, mape_p],
                    "RMSE": [rmse_d, rmse_p]
                })
                st.subheader("Model Accuracy (Backtest: June–Aug 2025)")
                st.dataframe(accuracy_table.round(3))





