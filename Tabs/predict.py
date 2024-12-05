import streamlit as st
from web import predict, load_data

def app(df, x, y):
    st.title('Prediksi Penyakit Kanker Paru')

    df, x, y = load_data()

    # Select model
    model_choice = st.selectbox("Choose a model", ["Decision Tree", "Naive Bayes", "SVM"])

    model_type_map = {"Decision Tree": "decision_tree", "Naive Bayes": "naive_bayes", "SVM": "svm"}
    model_type = model_type_map[model_choice]

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input('Age', min_value=0)  
        Gender = st.number_input('Gender', min_value=0)  
        Air_Pollution = st.number_input('Air Pollution', min_value=0)  
        Alcohol_use = st.number_input('Alcohol use', min_value=0)  
        Dust_Allergy = st.number_input('Dust Allergy', min_value=0)  
        Occupational_Hazards = st.number_input('Occupational Hazards', min_value=0)  
        Genetic_Risk = st.number_input('Genetic Risk', min_value=0)  
        Chronic_Lung_Disease = st.number_input('Chronic Lung Disease', min_value=0)  

    with col2:
        Balanced_Diet = st.number_input('Balanced Diet', min_value=0)  
        Obesity = st.number_input('Obesity', min_value=0)  
        Smoking = st.number_input('Smoking', min_value=0)  
        Passive_Smoker = st.number_input('Passive Smoker', min_value=0)  
        Chest_Pain = st.number_input('Chest Pain', min_value=0)  
        Coughing_of_Blood = st.number_input('Coughing of Blood', min_value=0)  
        Fatigue = st.number_input('Fatigue', min_value=0)  
        Weight_Loss = st.number_input('Weight Loss', min_value=0)  

    with col3:
        Shortness_of_Breath = st.number_input('Shortness of Breath', min_value=0)  
        Wheezing = st.number_input('Wheezing', min_value=0)  
        Swallowing_Difficulty = st.number_input('Swallowing Difficulty', min_value=0)  
        Clubbing_of_Finger_Nails = st.number_input('Clubbing of Finger Nails', min_value=0)  
        Frequent_Cold = st.number_input('Frequent Cold', min_value=0)  
        Dry_Cough = st.number_input('Dry Cough', min_value=0)  
        Snoring = st.number_input('Snoring', min_value=0)  

    features = [Age, Gender, Air_Pollution, Alcohol_use, Dust_Allergy, Occupational_Hazards, Genetic_Risk, Chronic_Lung_Disease, 
                Balanced_Diet, Obesity, Smoking, Passive_Smoker, Chest_Pain, Coughing_of_Blood, Fatigue, Weight_Loss, 
                Shortness_of_Breath, Wheezing, Swallowing_Difficulty, Clubbing_of_Finger_Nails, Frequent_Cold, Dry_Cough, Snoring]
    
    if st.button("Prediksi"):
        prediction, score = predict(x, y, features)
        st.info("Prediksi Sukses...")

        if prediction[0] == 0:
            st.write('Risiko kanker paru: Rendah (Class 0)')
        elif prediction[0] == 1:
            st.write('Risiko kanker paru: Sedang (Class 1)')
        else:
            st.write('Risiko kanker paru: Tinggi (Class 2)')

        st.write(f"Model yang digunakan memiliki tingkat akurasi {score * 100:.2f}%")
