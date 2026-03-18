import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("student_mat_cleaned.csv")

@st.cache_resource
def train_model(df):
    features = ['studytime', 'failures', 'absences', 'G1', 'G2', 'health', 'freetime', 'famrel']
    X = df[features]
    y = df['G3']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "r2": r2_score(y_test, pred),
        "mse": mean_squared_error(y_test, pred),
        "features": features
    }
    return model, metrics

df = load_data()
model, metrics = train_model(df)

st.title("🎓 Student Final Grade Prediction Using Machine Learning")
st.write(
    "Predict a student's final grade (G3) using academic and behavioral features "
    "with a Random Forest Regressor."
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Enter Student Details")
    studytime = st.slider("Study Time", 1, 4, 2)
    failures = st.slider("Past Failures", 0, 3, 0)
    absences = st.number_input("Absences", min_value=0, value=2)
    G1 = st.slider("1st Period Grade (G1)", 0, 20, 10)
    G2 = st.slider("2nd Period Grade (G2)", 0, 20, 10)
    health = st.slider("Health", 1, 5, 3)
    freetime = st.slider("Free Time", 1, 5, 3)
    famrel = st.slider("Family Relationship Quality", 1, 5, 3)

    if st.button("Predict Final Grade"):
        sample = [[studytime, failures, absences, G1, G2, health, freetime, famrel]]
        predicted_grade = model.predict(sample)[0]
        st.success(f"Predicted Final Grade (G3): {predicted_grade:.2f} / 20")

with col2:
    st.subheader("Model Performance")
    st.metric("R² Score", f"{metrics['r2']:.4f}")
    st.metric("Mean Squared Error", f"{metrics['mse']:.4f}")

    st.subheader("Feature Importance")
    importances = model.feature_importances_

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metrics["features"], importances)
    ax.set_title("Feature Importance")
    ax.set_xlabel("Features")
    ax.set_ylabel("Importance")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots(figsize=(12, 6))
corr = df.corr(numeric_only=True)
heatmap = ax2.imshow(corr, aspect='auto')
ax2.set_xticks(range(len(corr.columns)))
ax2.set_xticklabels(corr.columns, rotation=90)
ax2.set_yticks(range(len(corr.columns)))
ax2.set_yticklabels(corr.columns)
ax2.set_title("Correlation Heatmap")
fig2.colorbar(heatmap)
st.pyplot(fig2)

st.markdown("---")
st.markdown("### Tech Stack")
st.write("Python, Pandas, Scikit-learn, Matplotlib, Streamlit")

st.markdown("### Run Locally")
st.code("streamlit run app.py", language="bash")