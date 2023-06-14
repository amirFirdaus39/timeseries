import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import *
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_predict
import pmdarima as pm
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import grangercausalitytests
#import plotly.express as px
import altair as alt
warnings.filterwarnings("ignore")
#000A30
#102759
#streamlit run "d:/streamlit/1_OneFlow.py"
st.set_page_config(
    page_title="OneFlow Time Series",
    page_icon=":chart_with_upwards_trend:",
    layout="wide")
w = 500
h = 350

dfx = pd.read_excel('data/ap_latest.xlsx')
dfx['YM']=pd.to_datetime(dfx['START_DATE']).dt.strftime("%Y%m").apply(lambda x: x[2:])

def dataRep(rep):
    df = dfx.loc[dfx['REGION_EN_NAME']==rep]
    df = df.drop(columns={'REGION_EN_NAME','START_DATE'})
    df = df.set_index(['YM'])
    df = df.sort_index()
    df = df[:-1]
    df = df.rename(columns={"MOS":"MOS_COMPLETED"})
    df1 = df[['TOTAL_MOS', 'MOS_COMPLETED','SLA','SLA1','SLA2'
            #'SLA3'
            ]]
    df2 = df[['RATE','TIMELY_ADMISSION_RATE', 
            'JOB_CONTINUITY',
        'QC_TIMELY_ADMISSION_RATE', 'QC_TIMELY_COLLECTION_RATE',
        'QC_TIMELY_APPROVAL_RATE', 'QCL1_FTPR' 
        #'L2_TIMELY_REVIEW_RATE','L2_REJECTION_RATE'
        ]]
    return df, df1, df2
df, df1, df2 = dataRep('Asia Pacific Region')

# dfr = dfx.loc[~(dfx['REGION_EN_NAME']=='Asia Pacific Region')]
# dfr = dfr.drop(columns={'START_DATE'})

def limit(data,a=0.0001,b=1.00001):
    lim = np.log((data-a)/(b-data))
    return np.nan_to_num(lim)
    #return np.sqrt(data)

def invertLimit(data,b=1):
    inv = np.exp(data)/(np.exp(data)+b)
    for i in range(len(inv)):
        if inv[i] == 0.5:
            inv[i] = 0
    return np.nan_to_num(inv)
    #return np.square(data)

def fc_hwes(train, step, trend='add',season='add',period=3,title='HWES'):
    #transform = limit(train,a=0.0001, b=50000)
    history = [x for x in train]
    model = ExponentialSmoothing(history,trend=trend,seasonal=season,seasonal_periods=period)
    model_fit = model.fit()
    mae = round(mean_absolute_error(history, model_fit.fittedvalues),2)
    fc = []
    for i in range(1,step+1):
        model = ExponentialSmoothing(history,trend=trend,seasonal=season,seasonal_periods=period)
        model_fit = model.fit()
        newobs = model_fit.forecast(i)
        if newobs[0] < 0:
            newobs[0] = 0
        fc.append(newobs[0])
        history.append(newobs[0])
    #fc = invertLimit(fc,50000)
    idx = pd.date_range('2023-06-01','2024-12-01', freq='MS').strftime("%Y%m").to_series().apply(lambda x: x[2:])
    fc_df = pd.Series(fc)
    fc_df.index = idx[:step].values
    #train = train.append(fc_df)[:(len(train) + step)]
    col = {'Actual':train,'Forecast':fc_df}
    temp = pd.DataFrame(col)
    col1.subheader(f'Actual vs Forecast - {title}')
    col1.bar_chart(temp,width=w,height=h,use_container_width=True)
    col1.write(f"Forecast {title} Margin of error ± {mae}")
    #col1.table(df['Actual'])
    col1.table(temp['Forecast'][-step:])
    col1.divider()   
    return fc_df

def fc_hwes2(train, step, trend='add',season='add',period=3,title='TimeSeries'):
    transform = limit(train)
    history = [x for x in transform]
    model = ExponentialSmoothing(history,trend=trend,seasonal=season,seasonal_periods=period)
    model_fit = model.fit()
    mae = round(mean_absolute_error(train, invertLimit(model_fit.fittedvalues)),2)
    fc = []
    for i in range(1,step+1):
        model = ExponentialSmoothing(history,trend=trend,seasonal=season,seasonal_periods=period)
        model_fit = model.fit()
        newobs = model_fit.forecast(i)
        fc.append(newobs[0])
        history.append(newobs[0])
    fc = invertLimit(fc)
    idx = pd.date_range('2023-06-01','2024-12-01', freq='MS').strftime("%Y%m").to_series().apply(lambda x: x[2:])
    fc_df = pd.Series(fc)
    fc_df.index = idx[:step].values
    train = train.append(fc_df)[:(len(train) + step)]
    col = {'Actual':train,'Forecast':fc_df}
    temp = pd.DataFrame(col)
    if title == 'QCL1_FTPR':
        col1.subheader(f'Actual vs Forecast - {title}')
        col1.line_chart(temp,width=w,height=h,use_container_width=True)
        col1.write(f"Forecast {title} Margin of error ± {mae}")
        col1.table(temp['Forecast'][-step:])
        col1.divider()  
    else:
        col2.subheader(f'Actual vs Forecast - {title}')
        col2.line_chart(temp,width=w,height=h,use_container_width=True)
        col2.write(f"Forecast {title} Margin of error ± {mae}")
        col2.table(temp['Forecast'][-step:])
        col2.divider()  
    return fc_df

st.title(":chart_with_upwards_trend: OneFlow Time Series")
st.write("*Data Science Project by Amir Firdaus*")
url = "https://datafab-pro-id.gtsdata.huawei.com/DataFabKernelCn/#/visualBoardPreview?id=69Z07S0qg315oDPqTZ4gAo&questionPurpose=&isDesignPreview=true&pageId=pageld09pwn2"
st.write("Link to [One Flow Dashboard](%s)" % url)
tab1, tab2 = st.tabs(["Region", "Rep Office"])


with tab1:
    st.subheader(":earth_asia: One Flow Time Series [Asia Pacific]")
    colA, colB,colC = st.columns(3)
    n = colA.slider('Choose number of months to forecast:', min_value=1,max_value=5,value=3,key=1)
    st.divider()
    col1, col2 = st.columns(2)

    for col in df1.columns:
        fc_hwes(df1[col],n,title=col)

    for col in df2.columns:
        fc_hwes2(df2[col],n,title=col)

with tab2:
    st.subheader(":earth_asia: One Flow Time Series [Rep Office]")
    colA, colB,colC = st.columns(3)
    n2 = colA.slider('Choose number of months to forecast:', min_value=1,max_value=5,value=3,key=2)
    st.divider()
    col1, col2 = st.columns(2)
    repOff = colB.selectbox('Select Rep Office:',
                                ['Indonesia Rep Office',
                                 'Malaysia Rep Office',
                                'South Asia Rep Office',
                                'Thailand Rep Office',
                                'Japan Rep Office',
                                'Southeast Asia Multi-country Mgmt Dept',
                                'Singapore Rep Office',
                                'Philippines Rep Office',
                                'Hong Kong Rep Office'],
                                #default=['Malaysia Rep Office']
                                )
    dfR, dfA, dfB = dataRep(repOff)
    #st.write(dfB)
    for col in dfA.columns:
        fc_hwes(dfA[col],n2,title=col)

    for col in dfB.columns:
        fc_hwes2(dfB[col],n2,title=col)


    #st.write(options[0])

    # chart = alt.Chart(dfr).mark_line().encode(
    #     x=alt.X('YM:N'),
    #     y=alt.Y('RATE:Q'),
    #     color=alt.Color("REGION_EN_NAME:N")
    # ).properties(title="Hello World")
    # st.altair_chart(chart, use_container_width=True)
    # col1.line_chart(dfr[['REGION_EN_NAME','RATE']])
    # col1.table(dfr)
    #col1, col4 = st.columns(2)
