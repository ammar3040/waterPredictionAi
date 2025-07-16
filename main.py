import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
data = pd.read_csv("expanded_country_water_access_data.csv")
countries = sorted(data['Country'].unique())

# ----------- UI HEADER -----------
st.markdown("""
    <div style='text-align: center; padding: 10px; border-bottom: 2px solid #ccc;'>
        <h2 style='color: #4A90E2;'>🔮 Clean Water Access Prediction using AI</h2>
        <h4>Shaikh Ammar Irfan • Computer Engineering Sem 7</h4>
        <p>Shree Swami Atmanand Saraswati Institute of Technology | Institute Code: 076 | Branch Code: 7</p>
        <p>Enrollment Number: <strong>220760107118</strong></p>
    </div>
""", unsafe_allow_html=True)

st.write("🌍 Select a country to predict clean water access for **custom future years**.")

# ----------- Input: Country & Year Range -----------
selected_country = st.selectbox("Choose Country", countries)
year_range = st.slider("Select prediction year range", 2024, 2040, (2025, 2035), step=1)
future_years = np.arange(year_range[0], year_range[1]+1, 5).reshape(-1, 1)

# Filter & Predict
filtered = data[data['Country'] == selected_country]
X = filtered['Year'].values.reshape(-1, 1)
y = filtered['Water_Access_Percentage'].values
model = LinearRegression()
model.fit(X, y)
predictions = model.predict(future_years)

# ----------- Output Predictions -----------
st.subheader(f"📊 Predictions for {selected_country}")
for year, pred in zip(future_years.flatten(), predictions):
    st.markdown(f"- **{year}** → `{min(pred, 100):.2f}%`")

# ----------- Year to Reach 100% -----------
try:
    year_100 = int((100 - model.intercept_) / model.coef_[0])
    if year_100 <= 2040:
        st.success(f"🚀 {selected_country} may reach 100% access by **{year_100}**")
    else:
        st.warning("⚠️ 100% access may not be achieved by 2040.")
except:
    pass

# ----------- R² Score -----------
r2 = model.score(X, y)
st.markdown(f"📉 Model Confidence (R² Score): **{r2:.2f}**")

# ----------- Growth Rate -----------
if len(y) >= 2:
    growth = ((y[-1] - y[-2]) / y[-2]) * 100
    st.markdown(f"📈 Growth since last record: **{growth:.2f}%**")

# ----------- Graph ----------- 
fig, ax = plt.subplots()
ax.scatter(X, y, color='skyblue', label='Actual Data')
ax.plot(future_years, predictions, color='limegreen', marker='o', label='Prediction')
ax.set_xlabel("Year")
ax.set_ylabel("Water Access (%)")
ax.set_title(f"Water Access Trend – {selected_country}")
ax.set_ylim(0, 105)
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ----------- AI Suggestions -----------
st.subheader("💡 Suggestions to Improve Clean Water Access")
latest_access = y[-1]
if latest_access < 70:
    st.warning("🔸 Major efforts needed!")
    st.markdown("""
    - Launch rural water supply programs  
    - Promote rainwater harvesting  
    - Partner with NGOs for education  
    - Invest in waste treatment
    """)
elif 70 <= latest_access < 90:
    st.info("🔹 Moderate progress — needs scaling:")
    st.markdown("""
    - Expand piped systems  
    - Upgrade treatment plants  
    - Monitor quality in urban areas
    """)
else:
    st.success("✅ Strong progress — maintain excellence:")
    st.markdown("""
    - Use AI-powered sensors  
    - Run awareness campaigns  
    - Maintain digital monitoring
    """)

# ----------- Country Comparison -----------
st.subheader("🔍 Compare with Another Country (Optional)")
compare_country = st.selectbox("Compare With", ["None"] + [c for c in countries if c != selected_country])

if compare_country != "None":
    second = data[data['Country'] == compare_country]
    X2 = second['Year'].values.reshape(-1, 1)
    y2 = second['Water_Access_Percentage'].values
    model2 = LinearRegression()
    model2.fit(X2, y2)
    preds2 = model2.predict(future_years)

    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, color='skyblue', label=f'{selected_country} Actual')
    ax2.plot(future_years, predictions, color='green', label=f'{selected_country} Prediction')
    ax2.scatter(X2, y2, color='salmon', label=f'{compare_country} Actual')
    ax2.plot(future_years, preds2, color='red', label=f'{compare_country} Prediction')
    ax2.set_title(f"{selected_country} vs {compare_country}")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Water Access (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

# ----------- Historical Table -----------
st.subheader("📜 Historical Water Access")
st.dataframe(filtered[['Year', 'Water_Access_Percentage']].reset_index(drop=True))

# ----------- Top 10 Access Countries (2022) -----------
st.subheader("🌐 Top 10 Countries by Access (2022)")
latest = data[data['Year'] == 2022].sort_values(by="Water_Access_Percentage", ascending=False)
st.dataframe(latest[['Country', 'Water_Access_Percentage']].head(10))

# ----------- Download Predictions ----------- 
st.subheader("📥 Download Prediction Data")
download_df = pd.DataFrame({
    "Year": future_years.flatten(),
    "Predicted Access (%)": [min(p, 100) for p in predictions]
})
csv = download_df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", data=csv, file_name=f"{selected_country}_predictions.csv", mime="text/csv")

# ----------- Feedback ----------- 
st.subheader("📣 Feedback")
feedback = st.radio("How helpful is this prediction?", ["👍 Accurate", "👌 Okay", "👎 Not useful"])
if feedback:
    st.success("✅ Thank you for your feedback!")
