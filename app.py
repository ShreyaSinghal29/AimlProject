import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


st.title("🎓 Student Exam Pass/Fail Predictor")


df = pd.read_csv("student_exam_data_new.csv")
st.subheader("📊 Dataset")
st.write(df.head())


st.subheader("📈 Data Visualization")
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Study Hours", y="Previous Exam Score", hue="Pass/Fail", palette="coolwarm", ax=ax)
st.pyplot(fig)


X = df.drop("Pass/Fail", axis=1)
y = df["Pass/Fail"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("✅ Model Accuracy")
st.write(f"Accuracy: {accuracy:.2f}")

st.subheader("📋 Classification Report")
st.text(classification_report(y_test, y_pred))


st.subheader("🔍 Confusion Matrix")
fig2, ax2 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax2)
st.pyplot(fig2)


st.subheader("🎯 Make a Prediction")
study_hours = st.slider("Study Hours", 0.0, 10.0, 5.0)
previous_score = st.slider("Previous Exam Score", 0.0, 100.0, 50.0)

user_input = pd.DataFrame([[study_hours, previous_score]], columns=["Study Hours", "Previous Exam Score"])
user_input_scaled = scaler.transform(user_input)

prediction = model.predict(user_input_scaled)[0]
result = "✅ Will Pass" if prediction == 1 else "❌ Will Fail"
st.success(f"Prediction: {result}")
