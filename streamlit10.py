import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# --- Load Data (June, July, August only) ---
@st.cache_data
def load_data():
    june = "https://drive.google.com/uc?id=14g1GiSXFdg_tyWtRlXJbceEyWl3kvgmf"
    july = "https://drive.google.com/uc?id=17AX1AvTTD6-ri8ciZHaMrkSIcec7-Owz"
    aug  = "https://drive.google.com/uc?id=1N5wouJhhg_eF-6otEEoLxSTiJJz7jVaT"

    df_list = [
        pd.read_csv(june, low_memory=True),
        pd.read_csv(july, low_memory=True),
        pd.read_csv(aug,  low_memory=True),
    ]
    df = pd.concat(df_list, ignore_index=True)

    # lookup tables
    item_df = pd.read_csv("https://drive.google.com/uc?id=1pOxjGJOaWKDpQFWR_iKh7prSliQ6f_1X")
    premise_df = pd.read_csv("https://drive.google.com/uc?id=18c_Y_3JXF_G1S3LkUwscgNM743dfhO4f")

    item_df = item_df.drop_duplicates(subset="item_code")
    premise_df = premise_df.drop_duplicates(subset="premise_code")
    premise_df = premise_df[["premise_code", "premise", "state", "district"]]

    # merge
    df = df.merge(item_df, on="item_code", how="left")
    df = df.merge(premise_df, on="premise_code", how="left")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # keep only rows with valid dates and prices
    df = df.dropna(subset=["date", "price"])
    return df

df = load_data()

# --- Sidebar UI ---
st.sidebar.header("Filter Options")
state_list = df["state"].dropna().unique()
state_choice = st.sidebar.selectbox(
    "Select State",
    state_list,
    index=list(state_list).index("Selangor")  # default Selangor
)
district_choice = st.sidebar.selectbox(
    "Select District",
    df[df["state"] == state_choice]["district"].dropna().unique()
)
item_group_choice = st.sidebar.selectbox("Select Item Group", df["item_group"].dropna().unique())
item_category_choice = st.sidebar.selectbox(
    "Select Item Category",
    df[df["item_group"] == item_group_choice]["item_category"].dropna().unique()
)
item_choice = st.sidebar.selectbox(
    "Select Item",
    df[df["item_category"] == item_category_choice]["item"].dropna().unique()
)

premise_options = df[
    (df["state"] == state_choice) &
    (df["district"] == district_choice) &
    (df["item"] == item_choice)
]["premise"].dropna().unique()
premise_choice = st.sidebar.selectbox("Select Premise", premise_options)

forecast_date = st.sidebar.date_input(
    "Select Date to Forecast",
    min_value=pd.to_datetime("2025-09-01"),
    max_value=pd.to_datetime("2030-12-31")
)

# Extra validation: disallow Jan–Aug 2025
if forecast_date.year == 2025 and forecast_date.month < 9:
    st.warning("Please select a date from September 2025 onwards.")

run_button = st.sidebar.button("Run Forecast")

st.title("Malaysia Item Price Forecasting")

if run_button:
    # filter
    filtered = df[
        (df["state"] == state_choice) &
        (df["district"] == district_choice) &
        (df["item"] == item_choice)
    ].copy()

    if filtered.empty:
        st.warning("No data found for this selection.")
    else:
        # daily average
        daily = filtered.groupby("date", as_index=False)["price"].mean()

        # training = June 1 to Aug 31 (only from the loaded months)
        train_df = daily[(daily["date"] >= "2025-06-01") & (daily["date"] <= "2025-08-31")].copy()
        if train_df.empty or len(train_df) < 10:
            st.warning("Not enough training data (June–Aug).")
        else:
            # validate forecast_date is >= 2025-09-01
            min_test_date = pd.to_datetime("2025-09-01").date()
            sel_date = pd.to_datetime(forecast_date).date()
            if sel_date < min_test_date:
                st.warning("Please select a date on or after 2025-09-01 for the test period.")
            else:
                # fit Prophet
                df_prophet = train_df.rename(columns={"date": "ds", "price": "y"})
                model = Prophet()
                model.fit(df_prophet)

                # compute periods relative to LAST training date (robust!)
                last_train_date = df_prophet["ds"].max().date()
                periods = (sel_date - last_train_date).days
                if periods < 1:
                    st.warning(f"Training data goes up to {last_train_date}. No future periods to forecast to your selected date.")
                else:
                    # make future dataframe and predict
                    future = model.make_future_dataframe(periods=periods, freq="D")
                    forecast_df = model.predict(future)

                    # slice forecast from 2025-09-01 up to the selected date (inclusive)
                    start_dt = pd.to_datetime("2025-09-01")
                    end_dt = pd.to_datetime(sel_date)
                    forecast_slice = forecast_df[(forecast_df["ds"] >= start_dt) & (forecast_df["ds"] <= end_dt)][["ds", "yhat", "yhat_lower", "yhat_upper"]]

                    # Plot
                    fig, ax = plt.subplots(figsize=(12, 5))

                    # District average historical
                    ax.plot(daily["date"], daily["price"], label="Historical (Jun–Aug) – District Avg", marker="o",
                            markersize=3)

                    # District forecast
                    ax.plot(forecast_slice["ds"], forecast_slice["yhat"], "r-x",
                            label="Forecast (Sep onwards) – District Avg")

                    # Premise historical + forecast
                    if premise_choice:
                        # Historical
                        premise_data = filtered[filtered["premise"] == premise_choice].groupby("date", as_index=False)[
                            "price"].mean()
                        ax.plot(premise_data["date"], premise_data["price"], label=f"Historical – {premise_choice}",
                                marker="s", markersize=3)

                        # Forecast
                        train_premise = premise_data[
                            (premise_data["date"] >= "2025-06-01") & (premise_data["date"] <= "2025-08-31")].copy()
                        if not train_premise.empty and len(train_premise) >= 10:
                            df_prophet_premise = train_premise.rename(columns={"date": "ds", "price": "y"})
                            model_premise = Prophet()
                            model_premise.fit(df_prophet_premise)

                            last_train_date_premise = df_prophet_premise["ds"].max().date()
                            periods_premise = (sel_date - last_train_date_premise).days
                            if periods_premise >= 1:
                                future_premise = model_premise.make_future_dataframe(periods=periods_premise, freq="D")
                                forecast_premise = model_premise.predict(future_premise)
                                forecast_slice_premise = forecast_premise[
                                    (forecast_premise["ds"] >= start_dt) & (forecast_premise["ds"] <= end_dt)]
                                ax.plot(
                                    forecast_slice_premise["ds"],
                                    forecast_slice_premise["yhat"],
                                    "g--", label=f"Forecast – {premise_choice}"
                                )

                    ax.set_title(f"Prophet Forecast for {item_choice} in {district_choice}, {state_choice}")
                    ax.set_ylabel("Average Price (RM)")
                    ax.legend()
                    st.pyplot(fig)

                    # --- MESSAGE ---
                    selected_forecast = forecast_slice[forecast_slice["ds"] == end_dt]["yhat"].values
                    if len(selected_forecast) > 0:
                        forecast_value = selected_forecast[0]
                        st.markdown(
                            f"""
                            <div style='border:1px solid #ddd; padding:10px; border-radius:8px; background-color:#f9f9f9;'>
                                <span style='font-size:14px;'>
                                Prophet's Forecast of Average Price of <b>{item_choice}</b> 
                                in <b>{district_choice}, {state_choice}</b> on <b>{sel_date}</b>: 
                                RM <b>{forecast_value:.2f}</b>
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                    # --- Premise Forecast Message ---
                    if premise_choice and not train_premise.empty and len(train_premise) >= 10:
                        # get premise forecast value for selected date
                        premise_forecast_value = forecast_slice_premise[forecast_slice_premise["ds"] == end_dt][
                            "yhat"].values
                        if len(premise_forecast_value) > 0:
                            premise_value = premise_forecast_value[0]
                            # calculate percentage difference
                            pct_diff = ((premise_value - forecast_value) / forecast_value) * 100
                            diff_text = f" ({abs(pct_diff):.2f}% {'higher' if pct_diff > 0 else 'lower'} than district average)"

                            st.markdown(
                                f"""
                                <div style='border:1px solid #ddd; padding:10px; border-radius:8px; background-color:#eef9ff;'>
                                    <span style='font-size:14px;'>
                                    Prophet's Forecast for <b>{item_choice}</b> 
                                    at <b>{premise_choice}</b> on <b>{sel_date}</b>:  
                                    RM <b>{premise_value:.2f}</b> {diff_text}
                                    </span>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    # --- Prepare District Forecast Table ---
                    district_table = forecast_slice[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                    district_table = district_table.sort_values("ds").reset_index(drop=True)
                    district_table_display = district_table.rename(
                        columns={
                            "yhat": "Forecast (District)",
                            "yhat_lower": "Lower (District)",
                            "yhat_upper": "Upper (District)"
                        }
                    ).set_index("ds")

                    # --- Prepare Premise Forecast Table ---
                    if premise_choice and not train_premise.empty and len(train_premise) >= 10:
                        premise_table = forecast_slice_premise[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
                        premise_table = premise_table.sort_values("ds").reset_index(drop=True)

                        # calculate % difference vs district
                        premise_table["Change (%)"] = ((premise_table["yhat"] - district_table_display[
                            "Forecast (District)"].values)
                                                       / district_table_display["Forecast (District)"].values) * 100

                        premise_table_display = premise_table.rename(
                            columns={
                                "yhat": f"Forecast ({premise_choice})",
                                "yhat_lower": f"Lower ({premise_choice})",
                                "yhat_upper": f"Upper ({premise_choice})"
                            }
                        ).set_index("ds")
                    else:
                        premise_table_display = pd.DataFrame()

                    # --- Combine Tables ---
                    if not premise_table_display.empty:
                        combined_table = pd.concat(
                            [district_table_display, premise_table_display[["Forecast (" + premise_choice + ")",
                                                                            "Lower (" + premise_choice + ")",
                                                                            "Upper (" + premise_choice + ")",
                                                                            "Change (%)"]]], axis=1)
                    else:
                        combined_table = district_table_display

                    st.subheader(f"Forecasted Prices: 2025-09-01 → {sel_date}")
                    st.dataframe(combined_table.round(2))

                    # district backtest
                    y_true = df_prophet["y"].values
                    y_pred = model.predict(df_prophet[["ds"]])["yhat"].values
                    mae_d = mean_absolute_error(y_true, y_pred)
                    mape_d = (abs(y_true - y_pred) / y_true).mean() * 100
                    rmse_d = np.sqrt(mean_squared_error(y_true, y_pred))

                    # premise backtest (if enough data)
                    if not train_premise.empty and len(train_premise) >= 10:
                        df_prophet_premise = train_premise.rename(columns={"date": "ds", "price": "y"})
                        y_true_p = df_prophet_premise["y"].values
                        y_pred_p = model_premise.predict(df_prophet_premise[["ds"]])["yhat"].values
                        mae_p = mean_absolute_error(y_true_p, y_pred_p)
                        mape_p = (abs(y_true_p - y_pred_p) / y_true_p).mean() * 100
                        rmse_p = np.sqrt(mean_squared_error(y_true_p, y_pred_p)) if y_true_p is not None else None
                    else:
                        mae_p = mape_p = rmse_p = None

                    # create table
                    accuracy_table = pd.DataFrame({
                        "Model": ["District Avg", premise_choice if premise_choice else "N/A"],
                        "MAE": [mae_d, mae_p],
                        "MAPE (%)": [mape_d, mape_p],
                        "RMSE": [rmse_d, rmse_p]
                    })

                    st.subheader("Model Accuracy (Backtest: June–Aug 2025)")
                    st.dataframe(accuracy_table.round(3))

