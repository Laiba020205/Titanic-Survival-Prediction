import streamlit as st
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("train.csv")


# -----------------------------
# Prepare data
# -----------------------------
features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

X = df[features]
y = df["Survived"]


# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Preprocessing
# -----------------------------
numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# -----------------------------
# Logistic Regression model
# -----------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

model.fit(X_train, y_train)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚢 Titanic Survival Predictor")

st.write(
    "Enter passenger information to predict whether "
    "the passenger would have survived."
)

st.divider()


pclass = st.selectbox(
    "Passenger Class",
    [1, 2, 3]
)

sex = st.selectbox(
    "Sex",
    ["female", "male"]
)

age = st.number_input(
    "Age",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)

sibsp = st.number_input(
    "Number of Siblings/Spouses",
    min_value=0,
    max_value=10,
    value=0
)

parch = st.number_input(
    "Number of Parents/Children",
    min_value=0,
    max_value=10,
    value=0
)

fare = st.number_input(
    "Fare",
    min_value=0.0,
    value=30.0
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["S", "C", "Q"]
)


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Survival"):

    passenger = pd.DataFrame({
        "Pclass": [pclass],
        "Sex": [sex],
        "Age": [age],
        "SibSp": [sibsp],
        "Parch": [parch],
        "Fare": [fare],
        "Embarked": [embarked]
    })

    prediction = model.predict(passenger)[0]
    probability = model.predict_proba(passenger)[0]

    st.divider()

    if prediction == 1:
        st.success("Prediction: Survived")
    else:
        st.error("Prediction: Did Not Survive")

    st.write(
        f"Probability of survival: **{probability[1] * 100:.2f}%**"
    )