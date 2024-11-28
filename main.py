#Import Library
import streamlit as st
from web import load_data

from Tabs import home, predict, visualise

Tabs = {
    "Home" : home,
    "Prediction" : predict,
    "Visualisation" : visualise
}

#Sidebar
st.sidebar.title("Navigasi")

#Radio Option
page = st.sidebar.radio("Pages", list(Tabs.keys()))

#Load Dataset
df, x, y = load_data()

#Call App Function

if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df,x,y)
else:
    Tabs[page].app()