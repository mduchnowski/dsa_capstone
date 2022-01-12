# M.Duchnowski
# Upskill FY2022 : Data Science Academy
# Capstone Project
# 01/20/2022


# --- import libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import datetime
import json
import plotly.io as pio
pio.templates.default="ggplot2"


# --- define constants

today = datetime.date.today()


def load_data():
    df = pd.read_csv("data/data_capstone_dsa2021_2022.csv")
    df['cohort'] = pd.qcut(df['sum_score'], 3, labels=['Low', 'Mid', 'High'])

    #states = df.state.unique().tolist()
    #states.sort()

    #state_fips = pd.read_csv('data/state_fips.csv')
    #state_fips_dict = dict(zip(state_fips.state, state_fips.fips))
    
    return df

df = load_data()

st.title("Welcome to Matt's Data Science Project")

st.markdown("---")
st.markdown("## Dynamics matters")
st.markdown("Newton's second law of motion, _**F**_ = _**m**_ _**a**_, links the apparent position changes of an object to the force that leads to these changes, making it possible to undertand the underlying mechanism of the system and therefore predict the future of the changes. Here we draw a simple analogy, trying to understand the _**force**_ that drives the change of the numbers of COVID cases.")

st.markdown("The widely used [compartmental models in epidemiology](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology), such as the SRI model, are all based on the first order derivative of the number of cases and therefore they describe more about the _**kinematics**_. The ideas here are to consider the second order derivative to try to shed some light on the _**dynamics**_ of the changes of the numbers." )

st.markdown("The data used in this site is from [NY Times COVID-19 data](https://github.com/nytimes/covid-19-data). Please send your comments and suggestions to: [jianganghao@gmail.com](jianganghao@gmail.com)")
st.markdown("---")


st.markdown("## Plot of item timing by performace")

TimePlot = px.scatter(df, x="rt_gs_2", y="gs_2", size="sum_score", color="cohort", hover_name="age", log_x=True, size_max=60)

st.plotly_chart(TimePlot)
