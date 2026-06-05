import streamlit as st
import pandas as pd
import numpy as np
from statistics import mode
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    cross_val_score, 
    StratifiedKFold,
    train_test_split
)

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
)

from imblearn.over_sampling import RandomOverSampler

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Disease Prediction System", 
    layout="wide"
)

# ---------------- TITLE ---------------
st.title("Disease Prediction System")

st.markdown("Welcome to the Disease Prediction System – a simple and interactive tool to analyze health data and predict diseases using machine learning.")

# ---------------- DATA UPLOAD ----------------
uploaded_file = st.sidebar.file_uploader(
    "Upload dataset CSV",
    type=["csv"]
)

# Stop until user uploads dataset
if uploaded_file is None:
    st.info("Please upload a dataset CSV file from the sidebar to continue.")
    st.stop()

# Read uploaded dataset
data = pd.read_csv(uploaded_file)
df = data.copy()


# ---------------- SIDEBAR OPTIONS ----------------
st.sidebar.markdown("---")

st.sidebar.subheader("Preprocessing options")

resample = st.sidebar.checkbox(
    "Apply RandomOverSampler (handle class imbalance)", 
    value=True
)

n_splits = st.sidebar.number_input(
    "CV folds (StratifiedKFold)", 
    min_value=2, 
    max_value=10, 
    value=3
)

show_plots = st.sidebar.checkbox(
    "Show class distribution plot", 
    value=True
)



# ---------------- CHECK TARGET COLUMN ----------------
if 'disease' not in df.columns:
    st.error("Dataset must contain a column named 'disease'.")
    st.stop()


# ---------------- LABEL ENCODING ----------------
encoder = LabelEncoder()

df['disease_encoded'] = encoder.fit_transform(df['disease'])


# ---------------- FEATURES & TARGET ----------------
X = df.drop(columns=['disease', 'disease_encoded'])
y = df['disease_encoded']

# Encode gender if available
if 'gender' in X.columns:
    gender_encoder = LabelEncoder()
    X['gender'] = gender_encoder.fit_transform(
        X['gender'].astype(str)
    )

# Fill missing values
X = X.fillna(0)


# ---------------- DATASET PREVIEW ----------------
st.subheader("Dataset Preview")

st.write(f"Rows: {df.shape[0]}")

st.write(f"Features: {X.shape[1]}")

st.dataframe(df.head())

# ---------------- CLASS DISTRIBUTION ----------------
if show_plots:
    st.subheader("Class Distribution")

    fig, ax = plt.subplots(figsize=(22, 6))  # Wider figure to avoid crowding
    sns.countplot(
        data=df, 
        x="disease_encoded", 
        ax=ax
    )

    ax.set_title("Class Counts")
    
    plt.xticks(rotation=90)  # Rotate labels so they don't overlap
    
    plt.tight_layout()

    st.pyplot(fig)


# -------- RESAMPLING --------
if resample:

    ros = RandomOverSampler(random_state=42)
    
    X_resampled, y_resampled = ros.fit_resample(X, y)

else:

    X_resampled = X.copy()
    y_resampled = y.copy()

st.write(f"After resampling Rows: {X_resampled.shape[0]}")

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

# -------- MODEL SELECTION --------
st.sidebar.markdown("---")

st.sidebar.subheader("Models")

use_dt = st.sidebar.checkbox("Decision Tree", True)

use_rf = st.sidebar.checkbox("Random Forest", True)

use_nb = st.sidebar.checkbox("Naive Bayes", True)

use_knn = st.sidebar.checkbox("KNN (k=5)", True)

models = {}

if use_dt: 
    models["Decision Tree"] = DecisionTreeClassifier()

if use_rf: 
    models["Random Forest"] = RandomForestClassifier(random_state=42)

if use_nb: 
    models["Naive Bayes"] = GaussianNB()

if use_knn: 
    models["KNN"] = KNeighborsClassifier(n_neighbors=5)

if len(models) == 0:
    st.sidebar.warning("Select at least one model.")

# ---------------- TRAIN & EVALUATE ----------------
if st.sidebar.button("Train & Evaluate"):

    with st.spinner("Training models..."):

        skf = StratifiedKFold(
            n_splits=int(n_splits), 
            shuffle=True, 
            random_state=42
        )

        st.subheader("Cross-Validation Results")
        
        preds = {}
        
        trained_models = {}

        metrics_list = []

        # -------- MODEL TRAINING --------
        for name, model in models.items():

            scores = cross_val_score(
                model, 
                X_train, 
                y_train, 
                cv=skf, 
                scoring="accuracy"
            )
            
            st.write(
                f"**{name}** → Scores: "
                f"{np.round(scores,4)} "
                f"| Mean Accuracy: {scores.mean():.4f}"
            )

            # Train model
            model.fit(X_train, y_train)

            trained_models[name] = model
            
            # Prediction on test data
            y_pred = model.predict(X_test)

            preds[name] = y_pred

            # Metrics
            acc = accuracy_score(y_test, y_pred)

            prec = precision_score(
                y_test,
                y_pred,
                average='weighted',
                zero_division=0
            )

            rec = recall_score(
                y_test,
                y_pred,
                average='weighted',
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average='weighted',
                zero_division=0
            )

            metrics_list.append([
                name,
                acc,
                prec,
                rec,
                f1
            ])

        # ---------- METRICS TABLE ----------
        st.subheader("Metrics Summary Table")

        metrics_df = pd.DataFrame(
            metrics_list,
            columns=[
                "Model", 
                "Accuracy", 
                "Precision", 
                "Recall", 
                "F1-Score"
            ]
        )

        st.dataframe(
            metrics_df.style.format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}"
            })
        )

        best_model = metrics_df.loc[
            metrics_df["Accuracy"].idxmax()
        ]

        st.success(
            f"Best Performing Model: "
            f"{best_model['Model']} "
            f"with Accuracy "
            f"{best_model['Accuracy']:.4f}"
        )

        # ---------------- ACCURACY GRAPH ----------------
        st.subheader("Model Accuracy Comparison")

        fig2, ax2 = plt.subplots(figsize=(8, 5))

        sns.barplot(
            x="Model",
            y="Accuracy",
            data=metrics_df,
            ax=ax2
        )

        plt.xticks(rotation=15)

        st.pyplot(fig2)

        # ---------- COMBINED MODEL ----------
        if len(preds) >= 2:

            st.subheader("Combined Model (Majority Vote)")

            combined_preds = [
                mode(values) 
                for values in zip(*preds.values())
            ]

            acc = accuracy_score(
                y_test, 
                combined_preds
            )

            prec = precision_score(
                y_test, 
                combined_preds, 
                average='weighted', 
                zero_division=0
            )

            rec = recall_score(
                y_test, 
                combined_preds, 
                average='weighted', 
                zero_division=0
            )

            f1 = f1_score(
                y_test, 
                combined_preds, 
                average='weighted', 
                zero_division=0
            )

            st.write(
                f"Accuracy: {acc*100:.2f}% | "
                f"Precision: {prec:.4f} | "
                f"Recall: {rec:.4f} | "
                f"F1 Score: {f1:.4f}"
            )

        # Save models
        st.session_state["trained_models"] = trained_models

        st.session_state["X_columns"] = list(X.columns)

        st.session_state["encoder_classes"] = (
            encoder.classes_.tolist()
        )

        st.success("Training Completed Successfully!")

# ----------- PREDICTION SECTION------------
st.sidebar.markdown("---")

st.sidebar.subheader("Predict Disease")

if "trained_models" not in st.session_state:

    st.sidebar.info("Train models first.")

else:

    features = st.session_state["X_columns"]

    if len(features) <= 40:

        selected = st.sidebar.multiselect(
            "Select Symptoms", 
            options=features
        )

        if st.sidebar.button("Predict"):
        
            inp = [
                1 if col in selected else 0 
                for col in features
            ]

            input_df = pd.DataFrame(
                [inp], 
                columns=features
            )

            preds_out = {}

            for name, model in st.session_state["trained_models"].items():

                prediction = model.predict(input_df)[0]

                disease_name = (
                    st.session_state["encoder_classes"][prediction]
                )

                # Confidence
                confidence = 0
                
                if hasattr(model, "predict_proba"):

                    proba = model.predict_proba(input_df)[0]
                    
                    confidence = np.max(proba) * 100

                preds_out[name] = disease_name

                st.sidebar.write(
                    f"{name}: "
                    f"{disease_name} "
                    f"({confidence:.2f}%)"
                )

            # Majority vote
            final_vote = mode(list(preds_out.values()))

            st.sidebar.markdown(
                f"## Final Prediction: {final_vote}"
            )

    else:

        st.sidebar.write(
            "Too many features for manual selection."
        )

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown("Developed using Streamlit and Machine Learning")
