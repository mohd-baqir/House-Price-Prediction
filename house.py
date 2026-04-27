import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------
# Load datasets
# -------------------------------
data1 = pd.read_csv("data1.csv")
data2 = pd.read_csv("data2.csv")

# Combine datasets
data = pd.concat([data1, data2], ignore_index=True)

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# Rename columns (standardize)
data = data.rename(columns={
    'total_rooms': 'area',
    'total_bedrooms': 'bedrooms',
    'median_house_value': 'price',
    'bedroom': 'bedrooms',
    'bathroom': 'bathrooms',
    'space': 'area',
    'room': 'area'
})

# Remove duplicate columns
data = data.loc[:, ~data.columns.duplicated()]

# -------------------------------
# Title
# -------------------------------
st.title("🏠 House Price Prediction Dashboard")

# -------------------------------
# Dataset Preview
# -------------------------------
st.header("📄 Dataset Preview")
st.dataframe(data.head())

# -------------------------------
# Graph Section
# -------------------------------
st.header("📊 Graph Analysis")

if 'area' in data.columns and 'price' in data.columns:

    clean_data = data[['area', 'price']].dropna()

    option = st.selectbox(
        "Select Graph",
        ["Area vs Price (Scatter)", "Price Distribution", "Bedrooms vs Avg Price"]
    )

    # 1. Scatter Plot
    if option == "Area vs Price (Scatter)":
        sample_data = clean_data.sample(n=200)
        fig = plt.figure(figsize=(8,5))
        plt.scatter(sample_data['area'], sample_data['price'])
        plt.xlabel("Area")
        plt.ylabel("Price")
        plt.title("Area vs Price")
        plt.grid()
        st.pyplot(fig)

    # 2. Histogram
    elif option == "Price Distribution":
        fig = plt.figure(figsize=(8,5))
        plt.hist(clean_data['price'], bins=15)
        plt.xlabel("Price")
        plt.ylabel("Count")
        plt.title("Price Distribution")
        plt.grid()
        st.pyplot(fig)

    # 3. Clean Bar Chart (FIXED)
    elif option == "Bedrooms vs Avg Price":
        if 'bedrooms' in data.columns:
            avg = data.groupby('bedrooms')['price'].mean()

            fig = plt.figure(figsize=(8,5))
            plt.bar(avg.index, avg.values, width=0.5)
            plt.xlabel("Bedrooms")
            plt.ylabel("Average Price")
            plt.title("Bedrooms vs Average Price")
            plt.grid(axis='y')
            st.pyplot(fig)
        else:
            st.warning("Bedrooms column not found")

else:
    st.error("❌ Required columns not found!")

# -------------------------------
# Model + Prediction
# -------------------------------
st.header("🔮 Price Prediction")

if all(col in data.columns for col in ['area', 'bedrooms', 'price']):

    model_data = data[['area', 'bedrooms', 'price']].dropna()

    X = model_data[['area', 'bedrooms']]
    y = model_data['price']

    model = LinearRegression()
    model.fit(X, y)

    # User Input
    area = st.number_input("Enter Area (sq ft)", min_value=0.0)
    bedrooms = st.number_input("Enter Bedrooms", min_value=0.0)

    if st.button("Predict Price"):
        prediction = model.predict([[area, bedrooms]])
        st.success(f"Estimated Price: {prediction[0]:,.2f}")

else:
    st.warning("⚠️ Required columns missing")