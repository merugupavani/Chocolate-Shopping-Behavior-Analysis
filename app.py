import streamlit as st
import pandas as pd
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler

# --- STEP 1: LOAD ASSETS ---
# Load your model, scaler, and the feature columns list you saved in Jupyter
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_columns = pickle.load(open("features.pkl", "rb"))

# --- STEP 2: APP TITLE ---
st.set_page_config(page_title="ChocoPredict", page_icon="🍫")
st.title("Chocolate Customer Behavior Predictor")
st.write("Enter the customer details below to predict their spending category.")

# --- STEP 3: USER INPUTS (UI) ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 10, 80, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    category = st.selectbox("Category", ["Chocolate", "Snacks", "Drinks"])
    size = st.selectbox("Size", ["Small", "Medium", "Large"])
    season = st.selectbox("Season", ["Summer", "Winter", "Spring", "Autumn"])
    review = st.slider("Review Rating", 1.0, 5.0, 4.5)

with col2:
    subscription = st.selectbox("Subscription", ["Yes", "No"])
    discount = st.selectbox("Discount Applied", ["Yes", "No"])
    promo = st.selectbox("Promo Code Used", ["Yes", "No"])
    previous = st.number_input("Previous Purchases", 0, 50, value=10)
    payment = st.selectbox("Payment Method", ["Cash", "Card", "UPI"])
    shipping = st.selectbox("Shipping Type", ["Standard", "Express"])
    frequency = st.selectbox("Frequency", ["Weekly", "Fortnightly", "Annually"])

# --- STEP 4: PREDICTION LOGIC ---
if st.button("Predict Spender Category"):
    
    # 1. Convert Inputs → Dictionary
    input_dict = {
        "Age": age,
        "Gender": gender,
        "Category": category,
        "Size": size,
        "Season": season,
        "Review_Rating": review,
        "Subscription_Status": subscription,
        "Discount_Applied": discount,
        "Promo_Code_Used": promo,
        "Previous_Purchases": previous,
        "Payment_Method": payment,
        "Shipping_Type": shipping,
        "Frequency_of_Purchases": frequency
    }

    # 2. Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # 3. Apply One-Hot Encoding (get_dummies)
    input_df_encoded = pd.get_dummies(input_df)

    # 4. Align with Training Columns (CRITICAL)
    # This ensures if a user didn't pick "Drinks", that column still exists as 0
    input_df_final = input_df_encoded.reindex(columns=feature_columns, fill_value=0)

    # 5. Scaling
    input_scaled = scaler.transform(input_df_final)

    # 6. Prediction
    prediction = model.predict(input_scaled)[0]

    # 7. Map Output to Meaning
    label_map = {
        0: "Low Spender",
        1: "Medium Spender",
        2: "High Spender"
    }

    # --- STEP 5: OUTPUT ---
    st.markdown("---")
    result_text = label_map.get(prediction, "Unknown")
    
    if result_text == "High Spender":
        st.success(f"### Prediction: {result_text}")
        st.balloons()
    elif result_text == "Medium Spender":
        st.info(f"### Prediction: {result_text}")
    else:
        st.warning(f"### Prediction: {result_text}")