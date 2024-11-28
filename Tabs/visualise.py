import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn import tree
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import streamlit as st

from web import train_model

def app(df, x, y):
    
    st.title("Visualisasi Prediksi Penyakit Kanker Paru")

    def plot_confusion_matrix(model, x, y, model_name):
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay.from_estimator(model, x, y, display_labels=['0', '1', '2'], ax=ax)
        disp.plot(cmap=plt.cm.Oranges)
        st.pyplot(fig)
        y_pred = model.predict(x)
        accuracy = accuracy_score(y, y_pred)
        st.write(f"Accuracy for {model_name} model: {accuracy * 100:.0f}%")
        st.text(f"Classification Report for {model_name}: \n{classification_report(y, y_pred)}")

    if st.checkbox("Plot Confusion Matrix (SVM)"):
        model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        model.fit(x, y)
        plot_confusion_matrix(model, x, y, "SVM")
        

    if st.checkbox("Plot Confusion Matrix (Naive Bayes)"):
        model = GaussianNB()
        model.fit(x, y)
        plot_confusion_matrix(model, x, y, "Naive Bayes")

    if st.checkbox("Plot Confusion Matrix (Decision Tree)"):
        model, score = train_model(x, y, model_type="decision_tree")
        plot_confusion_matrix(model, x, y, "Decision Tree")
