import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import streamlit as st

from web import train_model

def app(df, x, y):
    st.title("Visualisasi Prediksi Penyakit Kanker Paru")

    # Slider dengan rentang 0-100
    test_size_percent = st.slider("Pilih rasio data uji (%)", 0, 100, 20, step=5)
    test_size = test_size_percent / 100

    if test_size == 0:
        st.warning("Rasio data uji 0%: Semua data digunakan untuk pelatihan.")
        x_train, x_test, y_train, y_test = x, None, y, None
    elif test_size == 1:
        st.warning("Rasio data uji 100%: Semua data digunakan untuk pengujian.")
        x_train, x_test, y_train, y_test = None, x, None, y
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    if x_train is not None and y_train is not None:
        st.write(f"Data training: {len(y_train)}")
    if x_test is not None and y_test is not None:
        st.write(f"Data testing: {len(y_test)}")
    
    # Fungsi untuk mengevaluasi model dan menampilkan hasil
    def evaluate_model(model, x_train, y_train, x_test, y_test, model_name):
        train_accuracy = model.score(x_train, y_train) * 100
        st.write(f"Akurasi data training {model_name}: {train_accuracy:.2f}%")

        if x_test is not None and y_test is not None:
            y_pred = model.predict(x_test)
            test_accuracy = accuracy_score(y_test, y_pred) * 100
            st.write(f"Akurasi data testing {model_name}: {test_accuracy:.2f}%")

            # Plot confusion matrix
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=['0', '1', '2'], ax=ax)
            disp.plot(cmap=plt.cm.Oranges)
            st.pyplot(fig)
            st.text(f"Classification Report for {model_name}:\n{classification_report(y_test, y_pred)}")
        else:
            st.warning(f"Tidak ada data testing untuk {model_name}. Menampilkan hasil untuk data training.")

            fig, ax = plt.subplots()
            y_pred_train = model.predict(x_train)
            disp = ConfusionMatrixDisplay.from_estimator(model, x_train, y_train, display_labels=['0', '1', '2'], ax=ax)
            disp.plot(cmap=plt.cm.Oranges)
            st.pyplot(fig)
            st.text(f"Classification Report for {model_name} (Training Data):\n{classification_report(y_train, y_pred_train)}")

    # Model SVM
    if st.checkbox("Plot Confusion Matrix (SVM)"):
        if x_train is None or y_train is None:
            st.error("Tidak ada data training untuk melatih model.")
        else:
            model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            model.fit(x_train, y_train)
            evaluate_model(model, x_train, y_train, x_test, y_test, "SVM")

    # Model Naive Bayes
    if st.checkbox("Plot Confusion Matrix (Naive Bayes)"):
        if x_train is None or y_train is None:
            st.error("Tidak ada data training untuk melatih model.")
        else:
            model = GaussianNB()
            model.fit(x_train, y_train)
            evaluate_model(model, x_train, y_train, x_test, y_test, "Naive Bayes")

    # Model Decision Tree
    if st.checkbox("Plot Confusion Matrix (Decision Tree)"):
        if x_train is None or y_train is None:
            st.error("Tidak ada data training untuk melatih model.")
        else:
            model, score = train_model(x_train, y_train, model_type="decision_tree")
            evaluate_model(model, x_train, y_train, x_test, y_test, "Decision Tree")
