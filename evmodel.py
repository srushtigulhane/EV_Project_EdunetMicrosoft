
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# -----------------------------
# Load EV Sales Model
# -----------------------------
ev_model_path = "ev_model9.pkl"   # use relative paths
with open(ev_model_path, "rb") as f:
    ev_data = pickle.load(f)
    ev_model = ev_data["model"]
    ev_columns = ev_data["columns"]  # columns used during training

# Load EV dataset for trend charts
ev_df = pd.read_csv("EV_Dataset2.csv")
ev_df["EV_Sales_Quantity"] = pd.to_numeric(
    ev_df["EV_Sales_Quantity"].astype(str).str.replace(",", ""), errors="coerce"
)
ev_df = ev_df.dropna(subset=["Year", "State", "EV_Sales_Quantity"])
ev_df["EV_Sales_Quantity_Lakhs"] = ev_df["EV_Sales_Quantity"] / 100000
ev_df["State"] = ev_df["State"].str.strip()

# -----------------------------
# Load Charging Station Model
# -----------------------------
cs_model_path = "cs_model.pkl"
with open(cs_model_path, "rb") as f:
    ch_data = pickle.load(f)
    ch_model = ch_data["model"]
    ch_columns = ch_data["columns"]

ch_df = pd.read_csv("final_dataset2.csv")
ch_df = ch_df.dropna(subset=["total-charging-stations", "State Name"])
ch_df["State Name"] = ch_df["State Name"].str.strip()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗💨 EV Sales & Charging Station Forecast")
mode = st.sidebar.radio("Mode", ["EV Sales", "Charging Stations"])

# ---------------- EV Sales Prediction ----------------
if mode == "EV Sales":
    year = st.sidebar.number_input(
        "Year",
        min_value=int(ev_df["Year"].min()),
        max_value=2050,
        value=int(ev_df["Year"].min()),
        step=1
    )
    state = st.sidebar.selectbox("State", sorted(ev_df["State"].unique()))
    state = state.strip()

    if st.sidebar.button("Predict EV Sales", key="ev_predict_btn"):
        # Prepare input for prediction
        input_df = pd.DataFrame({"Year": [year], "State": [state]})

        # One-hot encode (keep Year intact)
        input_encoded = pd.get_dummies(input_df, drop_first=False)

        # Align with training columns
        input_encoded = input_encoded.reindex(columns=ev_columns, fill_value=0)

        # Predict
        prediction = ev_model.predict(input_encoded)[0]
        st.success(f"Predicted EV Sales in {state} for {year}: **{prediction:.2f} Lakhs**")

        # Trend chart including predicted year
        state_sales = ev_df[ev_df["State"] == state].groupby("Year")["EV_Sales_Quantity_Lakhs"].sum()
        state_sales.loc[year] = prediction
        state_sales = state_sales.replace(0, 0.01).sort_index()
        st.line_chart(state_sales)

# --------------- Charging Station Prediction ---------------
elif mode == "Charging Stations":
    state = st.sidebar.selectbox("State", sorted(ch_df["State Name"].unique()))
    state = state.strip()

    # Input vehicle counts
    two_wheeler = st.sidebar.number_input("Two Wheeler", min_value=0, step=1000)
    three_wheeler = st.sidebar.number_input("Three Wheeler", min_value=0, step=1000)
    four_wheeler = st.sidebar.number_input("Four Wheeler", min_value=0, step=1000)
    goods_vehicle = st.sidebar.number_input("Goods Vehicles", min_value=0, step=1000)
    public_service = st.sidebar.number_input("Public Service Vehicle", min_value=0, step=100)
    special_category = st.sidebar.number_input("Special Category Vehicles", min_value=0, step=100)
    ambulance = st.sidebar.number_input("Ambulance/Hearses", min_value=0, step=10)
    construction = st.sidebar.number_input("Construction Equipment Vehicle", min_value=0, step=100)
    other = st.sidebar.number_input("Other Vehicles", min_value=0, step=100)
    grand_total = st.sidebar.number_input("Grand Total", min_value=0, step=1000)

    if st.sidebar.button("Predict Charging Stations"):
        input_df = pd.DataFrame([{
            "State Name": state,
            "Two Wheeler": two_wheeler,
            "Three Wheeler": three_wheeler,
            "Four Wheeler": four_wheeler,
            "Goods Vehicles": goods_vehicle,
            "Public Service Vehicle": public_service,
            "Special Category Vehicles": special_category,
            "Ambulance/Hearses": ambulance,
            "Construction Equipment Vehicle": construction,
            "Other": other,
            "Grand Total": grand_total
        }])

        # One-hot encode input (keep numeric intact)
        input_encoded = pd.get_dummies(input_df, drop_first=False)

        # Align with training columns
        input_encoded = input_encoded.reindex(columns=ch_columns, fill_value=0)

        # Predict & ensure positive
        prediction = max(int(ch_model.predict(input_encoded)[0]), 1)
        st.success(f"Predicted Charging Stations in {state}: **{prediction} stations**")


