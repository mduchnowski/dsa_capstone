# M.Duchnowski
# Upskill FY2022 : Data Science Academy
# Capstone Project
# 01/31/2022


# --- import libraries
import numpy as np
import pandas as pd
import datetime
import json
import re

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.figure_factory as ff
#import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import metrics

import streamlit as st

st.set_page_config(layout="wide")

#from sklearn import datasets, linear_model
#from sklearn.metrics import mean_squared_error, r2_score

pio.templates.default="ggplot2"

# --- define constants
today = datetime.date.today()

# Function for loading data into memory 
def load_data():
    
    df = pd.read_csv("data_capstone_dsa2021_2022.csv")
    
    # Calculate Item Stats - This includes only PPLUSES
    dfStats = df.filter(regex='^gs_(\d*)',axis=1).mean(axis=0).to_frame().T.add_suffix('_PPlus')
    
    #Every record will contain PPluses for easy access later
    df = df.merge(dfStats, how="cross")
    
    return df, dfStats

# Custom function for interpretting response to state 
@st.cache
def get_state(response):
    
    #https://gist.github.com/PhazeonPhoenix/2008449
    stateabbrev =  re.search(r"(\b)(A[LKSZRAEP]|C[AOT]|D[EC]|F[LM]|G[AU]|HI|I[ADLN]|K[SY]|LA|M[ADEHINOPST]|N[CDEHJMVY]|O[HKR]|P[ARW]|RI|S[CD]|T[NX]|UT|V[AIT]|W[AIVY])(\b)", response, re.IGNORECASE)
   
    #Dictionary assigning abbreviations from formal state name
    dictStates = {'Alabama' :'AL', 'Alaska' :'AK', 'Arizona' :'AZ', 'Arkansas' :'AR', 'California' :'CA', 'Colorado' :'CO', 'Connecticut' :'CT', 'Delaware' :'DE', 'Florida' :'FL', 'Georgia' :'GA', 'Hawaii' :'HI', 'Idaho' :'ID', 'Illinois' :'IL', 'Indiana' :'IN', 'Iowa' :'IA', 'Kansas' :'KS', 'Kentucky' :'KY', 'Louisiana' :'LA', 'Maine' :'ME', 'Maryland' :'MD', 'Massachusetts' :'MA', 'Michigan' :'MI', 'Minnesota' :'MN', 'Mississippi' :'MS', 'Missouri' :'MO', 'Montana' :'MT', 'Nebraska' :'NE', 'Nevada' :'NV', 'New Hampshire' :'NH', 'New Jersey' :'NJ', 'New Mexico' :'NM', 'New York' :'NY', 'North Carolina' :'NC', 'North Dakota' :'ND', 'Ohio' :'OH', 'Oklahoma' :'OK', 'Oregon' :'OR', 'Pennsylvania' :'PA', 'Rhode Island' :'RI', 'South Carolina' :'SC', 'South Dakota' :'SD', 'Tennessee' :'TN', 'Texas' :'TX', 'Utah' :'UT', 'Vermont' :'VT', 'Virginia' :'VA', 'Washington' :'WA', 'West Virginia' :'WV', 'Wisconsin' :'WI', 'Wyoming' :'WY'}

    #Iterate thru dictionary
    for key in dictStates:
        if re.search(key, response, re.IGNORECASE):
            return dictStates[key].upper()
    
    #Return abbreviation (if no full-name was found above) 
    if stateabbrev:
        return stateabbrev.group().upper()
    return ""

# Derive a list of student pairs based on their response pattern
# To be used in cheating analysis
@st.cache
def getSuspectPairs(df):
    dfA = df[['Pattern', 'abbrev', 'sum_score', 'zip_score', 'id']].rename(columns={"id": "idA"}).round({'zip_score': 1})
    dfB = df[['Pattern', 'abbrev', 'sum_score', 'zip_score', 'id']].rename(columns={"id": "idB"}).round({'zip_score': 1})
    dfPairs = dfA.merge(dfB).query('idA != idB')   
    #Only return pairs where state is known and students do not have a perfect score (only to make graph interesting)
    return dfPairs.loc[(dfPairs['abbrev'] != '') & (dfPairs['sum_score'] < 20),['idA','idB']]


# Derive some features of the data set
#    abbrev - the state bbreviation, based on string 'state'  (uses get_state function)
#    cohort - the percentile grouping based on total 'sum_score'
@st.cache
def derive_vars(df):
        
    #Create identifiers
    df["id"] = df.index
    
    #Create Response Pattern
    df['Pattern'] = df.filter(regex='^gs_([1-9]|1[0-9]|2[0])$',axis=1).apply(lambda x: ''.join(x.astype(int).astype('str')), axis=1)
           
    #Use custom function to extract state abbreviations
    df['abbrev'] = df['state'].apply(get_state)
    
    # Compute Zip Factor - the mulitplier needed for zip_score
    # Compute z-scores : https://stackoverflow.com/questions/24761998/pandas-compute-z-score-for-all-columns
    # Iterate over timing vars :  https://cmdlinetips.com/2019/04/how-to-select-columns-using-prefix-suffix-of-column-names-in-pandas/
    for col in list(df.filter(regex='^rt_gs_',axis=1).columns):
        col_zipfac = col + '_zipfactor'
       
        df[col_zipfac]= ((df[col] - df[col].mean())/df[col].std(ddof=0))
        df[col_zipfac] = 0 - df[col_zipfac].clip(lower=-3,upper=3)
        df[col_zipfac] = (df[col_zipfac] + 4)/3


    # Hardcode the zipfactor for first item
    df['rt_gs_1_zipfactor'] = 1

    #Apply zip factor to generate a zip_score
    for num in range(1,20 + 1):
        df['gs_' + str(num) + '_zip']=  df['gs_' + str(num)] * df['rt_gs_' + str(num) +'_zipfactor'] 
        if num == 1 :
            df['zip_score'] = df['gs_' + str(num)] * df['rt_gs_' + str(num) +'_zipfactor'] 
        else :
            df['zip_score'] = df['zip_score'] + df['gs_' + str(num)] * df['rt_gs_' + str(num) +'_zipfactor'] 

    #Assign students to cohort by quartile 
    df['sum_cohort'] = pd.qcut(df['sum_score'], 3, labels=['01-Low', '02-Mid', '03-High'])
      
    #Assign students to cohort by quartile 
    df['zip_cohort'] = pd.qcut(df['zip_score'], 3, labels=['01-Low', '02-Mid', '03-High'])
    
            
    return df

    
# Load our data - this includes student data as well as 
    # df      : dataframe containing student reponse data
    # dfStats : dataframe containing item statistics
df, dfStats = load_data()

# Derive features
df = derive_vars(df)

# Get matched records for cheating analysis
dfPairs = getSuspectPairs(df)

#Layout for Modeling Scatter-Plot 
lytScatter = go.Layout(
    xaxis=go.layout.XAxis(
       range=[0, 21],
       showgrid=True,
       zeroline=True,
       showline=True,
       gridcolor='#bdbdbd',
       gridwidth=2,
       zerolinecolor='#969696',
       zerolinewidth=4,
       linecolor='#636363',
       linewidth=1
    ),
    yaxis=go.layout.YAxis(
        range=[0,21],
        showgrid=True,
        zeroline=True,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=2,
        zerolinecolor='#969696',
        zerolinewidth=4,
        linecolor='#636363',
        linewidth=1
   ),
   height=400,
   width=400,
)

#Layout for Pacing Plot 
lytPacing = go.Layout(
    showlegend=False,
    xaxis=go.layout.XAxis(
       title_text = '',        
       range=[-1, 22],
       showgrid=False,
       zeroline=False,
       showline=True,
       gridcolor='#bdbdbd',
       gridwidth=1,
       zerolinecolor='#969696',
       zerolinewidth=4,
       linecolor='#636363',
       linewidth=1,
       tickvals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
       ticktext = ['Item 1', 'Item 2', 'Item 3', 'Item 4', 'Item 5', 'Item 6', 'Item 7', 'Item 8', 'Item 9', 'Item 10','Item 11', 'Item 12', 'Item 13', 'Item 14', 'Item 15', 'Item 16', 'Item 17', 'Item 18', 'Item 19', 'Item 20']
    ),
    yaxis=go.layout.YAxis(
        title_text = '',
        range=[0,3],
        showgrid=False,
        zeroline=False,
        showline=True,
        gridcolor='#bdbdbd',
        gridwidth=2,
        zerolinecolor='#969696',
        zerolinewidth=4,
        linecolor='#636363',
        linewidth=1,
        tickvals = [1, 2, 2.65 ],
        ticktext = ['<b>Student B</b>', '<b>Student A</b>', 'PPlus Value']
   ),
   height=400,
   width=1000,
)


#########################################################
# Exploration of Feature: Zip Score
#########################################################
st.header("Exploration of Feature: Zip Score")
st.markdown("This report is a capstone project for the 2021-22 Data Science Academy. Please send your comments to: [mduchnowski@ets.org](mduchnowski@ets.org)")

#########################################################
# Overview
#########################################################
st.markdown("---")
st.markdown("## Overview")
st.markdown("Traditionally, the sum of dichotomous items scores is used to produce a raw score.")

st.latex(r'''sum\_score = \sum_{i=1}^{n}x_{i}''')   


st.markdown("Here we will use timing data to produce an entirely new metric, the _**zip score**_ which measures correct responses that are issued quickly in comparison to other test-takers in the same administration. Using item-level timing data, a series of factors $y_{i}$ are generated for each test-taker where $y_{i}$ $\in [0,2]$. These factors are produced by calcualting z-scores for each item response time, clipping the distribution at $[-3\sigma,3\sigma$] and projecting the results onto $[0,2]$. As a result, we calculate our new metric:")

st.latex(r'''zip\_score = \sum_{i=1}^{n}x_{i}y_{i}''')   
            
st.markdown("This new feature will function as an item-score multiplier and will be used to produce a new metric, the _**zip score**_, which reflects how zippy a test-taker is: the ability to answer corectly _and_ quickly. ")



#########################################################
# Data Preparation
#########################################################
st.markdown("---")
st.markdown("## Data Preparation")
st.markdown("To prepare the data for these analyses, the following variables were derived.")

st.text("sum_score  : sum of item scores \nsum_cohort : classification of students according to their sum_score; Low, Mid, High cutoffs are drawn using top, middle and bottow thirds (quantiles)  \n\nzip_score  : sum of item zip scores \nzip_cohort : classification of students according to their zip scrore; Low, Mid, High cutoffs are drawn using top, middle and bottow thirds (quantiles)\n\nabbrev     : Formulation of US state abbreviations using student input. When a response can not be determined, it is left blank\n")

#########################################################
# Effects on Score Distribution
#########################################################
st.markdown("---")
st.markdown("## Effects on Score Distribution")
st.markdown('The new metric **_zip score_** impacts our ability to discriminate witin performance cohorts.')

colC1, colC2 = st.columns((5,5))


# Raw Score Distribution
figCohort1 = px.histogram(df, 
                          x="sum_score", nbins=20,  barmode='overlay', marginal="box", color="sum_cohort", 
                          category_orders={"sum_cohort": ["01-Low", "02-Mid", "03-High"]}, 
                          labels={"sum_cohort": "Raw Ability Cohort"}, 
                          title="<b>Raw Score Distribution</b>")

figCohort1.update_traces(opacity=0.75)
figCohort1.update_xaxes(type="linear", range=[0, 30])
figCohort1.update_layout(legend=dict(yanchor="top", y=0.60, xanchor="left", x=0.15))
colC1.plotly_chart(figCohort1)

# Zip Score Distribution
figCohort2 = px.histogram(df, 
                          x="zip_score", color="sum_cohort", marginal="box", barmode='overlay', 
                          category_orders={"sum_cohort": ["01-Low", "02-Mid", "03-High"]}, 
                          labels={"sum_cohort": "Raw Ability Cohort"}, 
                          title="<b>Zip Score Distribution</b>")

figCohort2.update_traces(opacity=0.75)
figCohort2.update_xaxes(type="linear", range=[0, 30])
figCohort2.update_layout(legend=dict(yanchor="top", y=0.60, xanchor="left", x=0.15))
colC2.plotly_chart(figCohort2)

#########################################################
# Mobility Between Cohorts
#########################################################
st.markdown("---")
st.markdown("## Mobility Between Cohorts")
st.markdown('Students with a high _raw score_ have more perterbations when calculating their _zip score_. As a result, $\sigma(z)$ varies directly with $x$.')

st.markdown('Students in "Mid"-ability cohort for _raw score_  experience the greatest shift in cohort to "High"-ability cohort for _zip score_.')

colD1, colD2 = st.columns((5,5))

boxPlot1 = px.box(df, 
                  x="sum_score", 
                  y="zip_score", 
                  color="sum_cohort", 
                  category_orders={"sum_cohort": ["01-Low", "02-Mid", "03-High"]},
                  labels={"sum_cohort": "Ability Cohort"},
                  title='<b>Classification of Cohorts<br>Raw Score and Zip Score</b>')

boxPlot1.add_hrect(y0=0, y1=df['zip_score'].quantile(q = 0.33), line_width=2, fillcolor="red", opacity=0.1)
boxPlot1.add_hrect(y0=df['zip_score'].quantile(q = 0.33), y1=df['zip_score'].quantile(q = 0.66), line_width=2, fillcolor="gold", opacity=0.1)
boxPlot1.add_hrect(y0=df['zip_score'].quantile(q = 0.66), y1=30, line_width=2, fillcolor="green", opacity=0.1)

boxPlot1.update_layout(legend=dict(yanchor="top", y=0.90, xanchor="left", x=0.05))

colD1.plotly_chart(boxPlot1)


lstCohort = ['01 - Low', '02 - Mid', '03 - High']

#Values for heatmap colors
z = pd.crosstab(df['zip_cohort'],df['sum_cohort'], dropna=False, normalize='all').values.tolist()

#Textual percentages to display in heatmap
z_text = []
for a in z:
    pctString = ['{:.2%}'.format(x) for x in a]
    z_text.append(pctString) 

heatMap1 = ff.create_annotated_heatmap(z,  
    x=lstCohort, 
    y=lstCohort,
    annotation_text=z_text, 
    colorscale="Greens",
    hoverinfo='z')
heatMap1.update_traces(text=z_text)
heatMap1.update_layout(title= "<b>Proportion of New Zip Cohort<br>By Original Raw Cohort</b>")
heatMap1.update_yaxes(title= "Zip Score Cohort", side="left")
heatMap1.update_xaxes(title= "Raw Score Cohort", side="bottom")

colD2.plotly_chart(heatMap1)


#########################################################
# Geographic Trends
#########################################################
st.markdown("---")
st.markdown("## Geographic Trends")

colG1, colG2, colG3 = st.columns((7,1,7))

colG1.markdown("It is suspected that gender and test location are a major influence on pacing. Students with a very large mean _**zip score**_ will  require closer examination. Browse the map below to determine which states you would like a state-specific analysis.") 

colE1, colE2, colE3,colE5 = st.columns((7,1,4,4))

#Descriptive stats by State
dfZipScoreByAbbrev = df.groupby("abbrev")["zip_score"].describe().reset_index()

figUSmap = go.Figure(data=go.Choropleth(
    locations = dfZipScoreByAbbrev['abbrev'],
    z  = dfZipScoreByAbbrev['mean'],
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Greens',
    colorbar_title = ""
))

figUSmap.update_layout(
    dragmode = False,
    title_text = "<b>Mean Zip Score by US State</b>",
    geo_scope='usa', # limit map scope to USA
)

colE1.plotly_chart(figUSmap, use_container_width=True)


opState = colG3.selectbox('Select a State for Report',dfZipScoreByAbbrev['abbrev'])

if opState:
    
    dfYes = df[(df["abbrev"] == opState)]
    dfNo  = df[(df["abbrev"] != opState)]
    
    dfYes_g = dfYes.groupby(['home_computer', 'gender']).size().reset_index()
    dfYes_g['percentage'] = dfYes.groupby(['home_computer', 'gender']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
    dfYes_g.columns = ['home_computer', 'gender', 'Counts', 'Percentage']

    figYes = px.bar(dfYes_g, x='home_computer', y=['Counts'], color='gender', labels={"home_computer": "Home Computer", "gender": "Gender", "value": "Count"}, title= "<b>" + opState + "</b>", text=dfYes_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))

    colE3.plotly_chart(figYes, use_container_width=True)

    dfNo_g = dfNo.groupby(['home_computer', 'gender']).size().reset_index()
    dfNo_g['percentage'] = dfNo.groupby(['home_computer', 'gender']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
    dfNo_g.columns = ['home_computer', 'gender', 'Counts', 'Percentage']

    figNo = px.bar(dfNo_g, x='home_computer', y=['Counts'], color='gender', labels={"home_computer": "Home Computer", "gender": "Gender", "value": "Count"}, title= "<b>~" + opState + "</b>", text=dfNo_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)))

    colE5.plotly_chart(figNo, use_container_width=True)

    
#########################################################
# Comparative Modeling
#########################################################
st.markdown("---")
st.markdown("## Comparative Modeling")
st.markdown("We will model the **_raw score_** using a subset of items. One model, $f(x)$ will simply be built on _raw item scores_ alone. The other model, $g(x,y,z)$, will be built using _raw item scores_ ($x$), _zip factors_ ($y$) and the interaction between the two ($z$).")

colA1, colA3, colA4 , colA5= st.columns((4,2,1,6))
col1, col2, x, col3, col4 , col5, y= st.columns((2,2,1,7,1,7,2))

# Would prefer to create these dynamically, but still learning how to handle the object on callback
chx01 = col1.checkbox('Item 01', value=1)
chx02 = col1.checkbox('Item 02', value=1)
chx03 = col1.checkbox('Item 03', value=1)
chx04 = col1.checkbox('Item 04', value=1)
chx05 = col1.checkbox('Item 05', value=1)
chx06 = col1.checkbox('Item 06', value=1)
chx07 = col1.checkbox('Item 07', value=1)
chx08 = col1.checkbox('Item 08', value=1)
chx09 = col1.checkbox('Item 09', value=1)
chx10 = col1.checkbox('Item 10', value=1)
chx11 = col2.checkbox('Item 11', value=1)
chx12 = col2.checkbox('Item 12', value=1)
chx13 = col2.checkbox('Item 13', value=1)
chx14 = col2.checkbox('Item 14', value=1)
chx15 = col2.checkbox('Item 15', value=1)
chx16 = col2.checkbox('Item 16', value=1)
chx17 = col2.checkbox('Item 17', value=1)
chx18 = col2.checkbox('Item 18', value=1)
chx19 = col2.checkbox('Item 19', value=1)
chx20 = col2.checkbox('Item 20', value=1)

#chxList = [False, chx01,chx02,chx03,chx04,chx05,chx06,chx07,chx08,chx09,chx10,chx11,chx12,chx13,chx14,chx15,chx16,chx17,chx18,chx19,chx20]
    

#Model A : parameter list
aList = []

#Model B : parameter list
bList = []
   
if chx01:
    aList.append('gs_1')  
    bList.extend(['gs_1', 'gs_1_zip', 'rt_gs_1_zipfactor'])  
if chx02:
    aList.append('gs_2')
    bList.extend(['gs_2', 'gs_2_zip', 'rt_gs_2_zipfactor'])  
if chx03:
    aList.append('gs_3')
    bList.extend(['gs_3', 'gs_3_zip', 'rt_gs_3_zipfactor'])  
if chx04:
    aList.append('gs_4')
    bList.extend(['gs_4', 'gs_4_zip', 'rt_gs_4_zipfactor'])   
if chx05:
    aList.append('gs_5')
    bList.extend(['gs_5', 'gs_5_zip', 'rt_gs_5_zipfactor'])   
if chx06:
    aList.append('gs_6')
    bList.extend(['gs_6', 'gs_6_zip', 'rt_gs_6_zipfactor'])   
if chx07:
    aList.append('gs_7')
    bList.extend(['gs_7', 'gs_7_zip', 'rt_gs_7_zipfactor'])
if chx08:
    aList.append('gs_8')
    bList.extend(['gs_8', 'gs_8_zip', 'rt_gs_8_zipfactor'])      
if chx09:
    aList.append('gs_9')
    bList.extend(['gs_9', 'gs_9_zip', 'rt_gs_9_zipfactor'])   
if chx10:
    aList.append('gs_10')
    bList.extend(['gs_10', 'gs_10_zip', 'rt_gs_10_zipfactor'])   
if chx11:
    aList.append('gs_11')  
    bList.extend(['gs_11', 'gs_11_zip', 'rt_gs_11_zipfactor'])   
if chx12:
    aList.append('gs_12')
    bList.extend(['gs_12', 'gs_12_zip', 'rt_gs_12_zipfactor'])      
if chx13:
    aList.append('gs_13')
    bList.extend(['gs_13', 'gs_13_zip', 'rt_gs_13_zipfactor'])   
if chx14:
    aList.append('gs_14')
    bList.extend(['gs_14', 'gs_14_zip', 'rt_gs_14_zipfactor'])     
if chx15:
    aList.append('gs_15')
    bList.extend(['gs_15', 'gs_15_zip', 'rt_gs_15_zipfactor'])   
if chx16:
    aList.append('gs_16')
    bList.extend(['gs_16', 'gs_16_zip', 'rt_gs_16_zipfactor'])   
if chx17:
    aList.append('gs_17')
    bList.extend(['gs_17', 'gs_17_zip', 'rt_gs_17_zipfactor'])   
if chx18:
    aList.append('gs_18')
    bList.extend(['gs_18', 'gs_18_zip', 'rt_gs_18_zipfactor'])    
if chx19:
    aList.append('gs_19')
    bList.extend(['gs_19', 'gs_19_zip', 'rt_gs_19_zipfactor'])   
if chx20:
    aList.append('gs_20')
    bList.extend(['gs_20', 'gs_20_zip', 'rt_gs_20_zipfactor'])          

# At this point we have constructed our list of independent variables 
# eg.
#
# aList = ['gs_3', 'gs_7', 'gs_9', 'gs_13', 'gs_17']
#
# bList = ['gs_3', 'gs_7', 'gs_9', 'gs_13', 'gs_17', 
#         'gs_3_zip', 'gs_7_zip', 'gs_9_zip', 'gs_13_zip', 'gs_17_zip', 
#         'rt_gs_3_zipfactor', 'rt_gs_7_zipfactor', 'rt_gs_9_zipfactor', 'rt_gs_13_zipfactor', 'rt_gs_17_zipfactor']


    
# Mathematical Expression of Models
colA3.latex(r'''f\begin{pmatrix}x\end{pmatrix}=\beta_{0}+\sum_{i=1}^{n}\beta_{i}\left (x_{i}  \right )''')   
colA5.latex(r'''g\begin{pmatrix}x,y,z\end{pmatrix}=\beta_{0}+\alpha _{0}+\gamma _{0}+\sum_{i=1}^{n}\beta_{i}\left (x_{i}  \right )+\sum_{i=1}^{n} \alpha_{i}\left (y_{i}  \right )+\sum_{i=1}^{n} \gamma_{i}\left(  z_{i}\right )''')


# Models currently hardcoded to predict sum_score
y = df['sum_score']

#Configure Models
Xa = df[aList]
Xa_train, Xa_test, ya_train, ya_test = train_test_split(Xa, y, test_size=0.2, random_state=0)

Xb = df[bList]
Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, y, test_size=0.2, random_state=0)

#Build Model A
regA = linear_model.LinearRegression()
regA.fit(Xa_train, ya_train)
ya_pred = regA.predict(Xa_test)

dfPredictionsA = pd.DataFrame({'Actual': ya_test, 'Prediction A': ya_pred})
dfPredictionsA['A'] = dfPredictionsA['Prediction A'].round(decimals = 0)
      
figA = px.scatter(dfPredictionsA, x="Actual", y="Prediction A", opacity = .2)
figA.update_layout(lytScatter)
figA.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
col3.plotly_chart(figA, use_container_width=True)

#Model A performance metric
aRMSE = '{0:.2f}'.format(np.sqrt(metrics.mean_squared_error(ya_test, ya_pred)))
col3.markdown("<h5 style='text-align: center;'>" + "Root-Mean-Square Error : " + aRMSE + "</h5>", unsafe_allow_html=True)

#Build Model B
regB = linear_model.LinearRegression()
regB.fit(Xb_train, yb_train)
yb_pred = regB.predict(Xb_test)

dfPredictionsB = pd.DataFrame({'Actual': yb_test, 'Prediction B': yb_pred})
dfPredictionsB['B'] = dfPredictionsB['Prediction B'].round(decimals = 0)

figB = px.scatter(dfPredictionsB, x="Actual", y="Prediction B", opacity = .2)
figB.update_layout(lytScatter)
figB.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')), selector=dict(mode='markers'))
col5.plotly_chart(figB, use_container_width=True)

#Model B performance metric
bRMSE = '{0:.2f}'.format(np.sqrt(metrics.mean_squared_error(yb_test, yb_pred)))
col5.markdown("<h5 style='text-align: center;'>" + "Root-Mean-Square Error : " + bRMSE + "</h5>", unsafe_allow_html=True)

#########################################################
# Zip Score as Fingerprint
#########################################################
st.markdown("---")
st.markdown("## Zip Score as Fingerprint")
st.markdown("A student's _zip score_ contains information about their pacing and might be used to detect pairs or groups of students with suspiciously improbable similarities in item timing. This illustration depicts the pacing of two students with identical response patterns and equivalent _zip scores_.")

colF1, colF2 = st.columns((2,8))

def makePacingPlot():
    #Draw random pair from the dfPairs
    dfDraw = dfPairs.sample()
    stuA = df[df['id'] == dfDraw.iat[0,0]]
    stuB = df[df['id'] == dfDraw.iat[0,1]]

    #No timing for Item 1, so set as a placeholder
    stuA['rt_gs_1'] = 25
    stuB['rt_gs_1'] = 25

    #Frame Rate Hardcoded
    FrameRate = 5

    frame_list = []

    #Initialize
    frame_list.append([0, 2, 0, "Student A"])
    frame_list.append([0, 1, 0, "Student B"])

    #The total number of frames the animation will contain
    FrameTotal = int(round((max(stuA['rt_total'].values[0],stuB['rt_total'].values[0])/FrameRate)+50, -2))+1

    #Student A 
    for Frame in range(1,FrameTotal):
        CumTimeA = 0
        OnItemA = 1
        if FrameRate*Frame > stuA['rt_total'].values[0]:
            frame_list.append([Frame, 2, 21, "Student A"])
        else:
            for OnItemA in range(1,20 + 1):
                TimeElapsed = FrameRate*Frame
                CumTimeA = CumTimeA + stuA['rt_gs_' + str(OnItemA)].values[0]
                if CumTimeA > TimeElapsed:
                    #Write Current Position
                    frame_list.append([Frame, 2, OnItemA, "Student A"])
                    break
                
    #Student B             
    for Frame in range(1,FrameTotal):  
        CumTimeB = 0
        OnItemB = 1
        if FrameRate*Frame > stuB['rt_total'].values[0]:
            frame_list.append([Frame, 1, 21, "Student B"])
        else:    
            for OnItemB in range(1,20 + 1):
                TimeElapsed = FrameRate*Frame        
                CumTimeB = CumTimeB + stuB['rt_gs_' + str(OnItemB)].values[0]
                if CumTimeB > TimeElapsed:
                    #Write Current Position
                    frame_list.append([Frame, 1, OnItemB, "Student B"])
                    break

    x = pd.DataFrame(frame_list, columns=['Frame', 'Student', 'Position', 'Type']) 

    pacingFig=px.scatter(x, 
                     y="Student", 
                     x="Position", 
                     animation_frame="Frame", color="Type", color_discrete_sequence=['yellow','blue'], range_x=(-1,22), range_y=(0,3),
                     title="")
    pacingFig.update_layout(lytPacing)

    #Draw right/wring squares
    for i in range(20):
        #Right/Wrong squares for Student A
        if stuA['gs_' + str(i+1)].values[0] == 1:
            pacingFig.add_vrect(x0=i+0.55, x1=i+1.46, y0=.58, y1=.75, line_width=2, fillcolor="green", opacity=0.4)
        else:
            pacingFig.add_vrect(x0=i+0.55, x1=i+1.46, y0=.58, y1=.75, line_width=2, fillcolor="red", opacity=0.4)
        #Right/Wrong squares for Student A    
        if stuB['gs_' + str(i+1)].values[0] == 1:
            pacingFig.add_vrect(x0=i+0.55, x1=i+1.46, y0=.25, y1=.42, line_width=2, fillcolor="green", opacity=0.4)
        else:
            pacingFig.add_vrect(x0=i+0.55, x1=i+1.46, y0=.25, y1=.42, line_width=2, fillcolor="red", opacity=0.4)


    pacingFig.add_trace(go.Scatter(
        x=list(range(1,20 + 1)),
        y=[2.75] * 20,
        mode="text",
        name="P+",
        text=dfStats.applymap(lambda x: '{0:.2f}'.format(x)).values.tolist()[0],
        textposition="bottom center"
    ))

    pacingFig.update_traces(marker=dict(size=17, line=dict(width=1, color='Black')), selector=dict(mode='markers'))

    colF2.plotly_chart(pacingFig, use_container_width=True) 

    colF1.markdown("<br><span style='font-size: 80%'><h5 style='text-align: left;'><b>State : </b>" + stuA['abbrev'].values[0] + "<br><b>Raw Score : </b>" + str(stuA['sum_score'].values[0]) + "<br><b>Zip Score : </b>" + "{:.1f}".format(stuA['zip_score'].values[0]) + "</h5></span>", unsafe_allow_html=True)
    
    colF1.markdown("<span style='font-size: 80%'><h5 style='text-align: left;'><b>Student A: </b>" + stuA['gender'].values[0] + ", Age " + str(stuA['age'].values[0]) + "</h5><br><b>Student B: </b>" + stuB['gender'].values[0] + ", Age " + str(stuB['age'].values[0]) + "</h5></span>", unsafe_allow_html=True)
        

    
if colF1.button('Select Suspicious Pair At Random'):
    makePacingPlot() 
