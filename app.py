import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv("stores_sales_forecasting.csv", encoding="latin1")

df.columns = df.columns.str.strip()

df['Order Date'] = pd.to_datetime(df['Order Date'])

df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month

monthly_sales = df.groupby('Month')['Sales'].sum().reset_index()

X = monthly_sales[['Month']]
y = monthly_sales['Sales']

model = LinearRegression()
model.fit(X, y)

st.title("Store Sales Forecasting Dashboard")

st.write("### Dataset Preview")
st.dataframe(df.head())

st.write("Monthly Sales Trend")

fig, ax = plt.subplots()
ax.plot(monthly_sales['Month'], monthly_sales['Sales'], marker='o')
ax.set_xlabel("Month")
ax.set_ylabel("Sales")

st.pyplot(fig)

st.write("Future Sales Prediction")

month = st.number_input("Enter Month (1-12)", 1, 12)

prediction = model.predict(pd.DataFrame([[month]], columns=['Month']))

st.success(f"Predicted Sales: {prediction[0]:.2f}")