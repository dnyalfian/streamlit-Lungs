import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import streamlit as st
from sklearn.metrics import classification_report

# Function to load data
@st.cache()
def load_data():
    # Load Data
    df = pd.read_csv('predict_datatransform_cancer_patient.csv')

    # Feature and target selection
    x = df[["Age", "Gender", "Air_Pollution", "Alcohol_use", "Dust_Allergy", "Occupational_Hazards", 
            "Genetic_Risk", "Chronic_Lung_Disease", "Balanced_Diet", "Obesity", "Smoking", "Passive_Smoker", 
            "Chest_Pain", "Coughing_of_Blood", "Fatigue", "Weight_Loss", "Shortness_of_Breath", "Wheezing", 
            "Swallowing_Difficulty", "Clubbing_of_Finger_Nails", "Frequent_Cold", "Dry_Cough", "Snoring"]]
    y = df[['Level']]

    return df, x, y

# Function to train models (Decision Tree, Naive Bayes, SVM)
@st.cache()
def train_model(x, y, model_type="decision_tree"):
    if model_type == "decision_tree":
        model = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'
        )
    elif model_type == "naive_bayes":
        model = GaussianNB()
    elif model_type == "svm":
        model = SVC(kernel='rbf', C=1.0,gamma='scale',random_state=42)

    model.fit(x, y)
    score = model.score(x, y)

    return model, score

# Function for prediction
def predict(x, y, features, model_type="decision_tree"):
    model, score = train_model(x, y, model_type)

    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(plt)

# Function to plot feature importance (Decision Tree)
def plot_feature_importance(model, x):
    if isinstance(model, DecisionTreeClassifier):
        feature_importance = model.feature_importances_
        features = x.columns
        plt.figure(figsize=(10, 6))
        sns.barplot(x=feature_importance, y=features)
        plt.title("Feature Importance - Decision Tree")
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        st.pyplot(plt)

# Function to compare model accuracies
def plot_accuracy_comparison(scores):
    models = list(scores.keys())
    accuracies = list(scores.values())
    plt.figure(figsize=(8, 6))
    sns.barplot(x=models, y=accuracies)
    plt.title("Model Accuracy Comparison")
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    st.pyplot(plt)

def display_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=['Low', 'Medium', 'High'])
    st.text(report)
