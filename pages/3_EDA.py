import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import *
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

def gen_heatmap(df):
    corrmat = df.corr(method='pearson')
    f, ax = plt.subplots(figsize=(3, 2))
    sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)
    return f

dfx = pd.read_excel('D:/ap_latest.xlsx')
dfx['YM']=pd.to_datetime(dfx['START_DATE']).dt.strftime("%Y%m").apply(lambda x: x[2:])
df = dfx.loc[dfx['REGION_EN_NAME']=="Asia Pacific Region"]
df = df.drop(columns={'REGION_EN_NAME','START_DATE'})
df.sort_values(by=['YM'], inplace=True)
df = df.set_index(['YM'])   
df = df[:-1]
df = df.rename(columns={'MOS':'MOS_COMPLETED'})
df = df[['TOTAL_MOS', 'MOS_COMPLETED','SLA','SLA1','SLA2','RATE','TIMELY_ADMISSION_RATE', 
            'JOB_CONTINUITY', 'QC_TIMELY_ADMISSION_RATE', 'QC_TIMELY_COLLECTION_RATE',
        'QC_TIMELY_APPROVAL_RATE', 'QCL1_FTPR']]
var = ['RATE','TOTAL_MOS', 'MOS_COMPLETED','SLA','SLA1','SLA2','TIMELY_ADMISSION_RATE', 
            'JOB_CONTINUITY', 'QC_TIMELY_ADMISSION_RATE', 'QC_TIMELY_COLLECTION_RATE',
        'QC_TIMELY_APPROVAL_RATE', 'QCL1_FTPR']

def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print (dfoutput)
    if dftest[1] <= 0.05:
        return "Strong evidence against the null hypothesis(p-value < 0.05), reject the null hypothesis. Data is stationary"
    else:
        return "Weak evidence against null hypothesis (p-value >= 0.05),indicating it is non-stationary"
    
def adf_test1(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    #print (kpss_output)
    if kpsstest[1] > 0.05:
        return "Strong evidence against the null hypothesis(p-value > 0.05), reject the null hypothesis. Data is stationary"
    else:
        return "Weak evidence against null hypothesis(p-value <= 0.05),indicating it is non-stationary"
    
def kpss_test1(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','#Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    return kpss_output

def eval(actual, forecast):
    mae = round(mean_absolute_error(actual, forecast),3)
    mape = round(mean_absolute_percentage_error(actual, forecast)*100,3)
    rmse = round(mean_squared_error(actual, forecast, squared=False),3)
    metric = {"MAE":mae,
              "MAPE":mape,
              "RMSE":rmse}
    eval_df = pd.DataFrame(metric,index=[0])
    return eval_df

def model(train,model_fit,col,model):
    fig = plt.figure(facecolor='#000a30')
    #plt.rcParams.update({'font.size': 8})
    ax = plt.axes()
    ax.set_facecolor("#000a30")
    ax.grid()
    ax.tick_params(labelcolor='white')
    plt.title(f'{model} {col}', fontsize=15,color='white')
    plt.plot(train, label='Actual')
    plt.plot(model_fit.fittedvalues[1:], label='Predict')
    plt.legend(loc='upper left', fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right', fontsize='medium')
    return fig

def gen_heatmap(df):
    plt.rcParams.update({'font.size': 10})
    corrmat = df.corr(method='pearson')
    fig = plt.figure(facecolor='#000a30')
    ax = plt.axes()
    ax.set_facecolor("#000a30")
    ax.tick_params(labelcolor='white')
    axx = sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.1)
    cbar = axx.collections[0].colorbar
    cbar.ax.tick_params(labelcolor='white')
    return fig

st.title(":mag: Exploratory Data Analysis - EDA")
tab1, tab2,tab3, tab4 = st.tabs(["Correlation","Stationarity","Seasonality/Trend","Model Fitting"])
with tab1:
    st.subheader("Correlation Matrix")
    col1,col2=st.columns(2)
    col1.markdown("     A correlation matrix is a statistical technique used to evaluate the relationship between two variables in a data set. The matrix is a table in which every cell contains a correlation coefficient, where 1 is considered a strong positive relationship between variables, 0 a neutral relationship and -1 a strong negative relationship")
    col1.markdown("     On the heatmap correlation display on the right, the darker the colour reflects a stronger positive relationship while the brighter colour reflects a stronger negative relationships.")
    col1.markdown("We can see the One Flow 'Rate' has strong positive relationship with 'Job Continuity' and 'QC_TIMELY_APPROVAL_RATE'; meaning the higher 'Job continuity' value, the higher the 'Rate' will be. While also having strong negative relationship with 'SLA', 'SLA1' and 'SLA2'; meaning lower SLA constitutes higher One Flow 'Rate'. ")
    col2.pyplot(gen_heatmap(df),use_container_width=True)
with tab2:  
    st.subheader("Stationarity")
    st1, st2 = st.columns(2)
    st_col = st1.selectbox("Choose variable to check stationarity:",options=var)
    ndiff = st2.slider('Choose number of differencing to apply:', min_value=1,max_value=3,value=1)
    st1.subheader(f"{st_col} from 2022-2023")
    st1.line_chart(df[st_col])
    t_adf, t_kpss = st2.tabs(['ADF Test','KPSS Test'])
    t_adf.table(adf_test1(df[st_col]))
    t_adf.markdown(adf_test(df[st_col]))
    t_kpss.table(kpss_test1(df[st_col]))
    t_kpss.markdown(kpss_test(df[st_col]))
    
    diff1 = df[st_col].diff().dropna()
    diff2 = df[st_col].diff().diff().dropna()
    diff3 = df[st_col].diff().diff().diff().dropna()
    t_adf2, t_kpss2 = st2.tabs(['ADF Test','KPSS Test'])
    if ndiff == 1:
        st1.subheader(f"{st_col} (1st Differencing) from 2022-2023")
        st1.line_chart(diff1)
        t_adf2.table(adf_test1(diff1))
        t_adf2.markdown(adf_test(diff1))
        t_kpss2.table(kpss_test1(diff1))
        t_kpss2.markdown(kpss_test(diff1))
    elif ndiff == 2:
        st1.subheader(f"{st_col} (2nd Differencing) from 2022-2023")
        st1.line_chart(diff2)
        t_adf2.table(adf_test1(diff2))
        t_adf2.markdown(adf_test(diff2))
        t_kpss2.table(kpss_test1(diff2))
        t_kpss2.markdown(kpss_test(diff2))
    else:
        st1.subheader(f"{st_col} (3rd Differencing) from 2022-2023")
        st1.line_chart(diff3)
        t_adf2.table(adf_test1(diff3))
        t_adf2.markdown(adf_test(diff3))
        t_kpss2.table(kpss_test1(diff3))
        t_kpss2.markdown(kpss_test(diff3))

with tab3:  
    st.subheader("Seasonality/Trend")
    st1, st2 = st.columns(2)
    ss_col = st1.selectbox("Choose variable to check seasonality/trend:",options=var)
    st1.write("")
    nperiod = st2.slider('Choose seasonality period to apply:', min_value=2,max_value=6,value=3)
    decompose_result = seasonal_decompose(df[ss_col],model='additive',period=nperiod)
    decompose_result_mul = seasonal_decompose(df[ss_col],model='multiplicative',period=nperiod)
    #st1.divider()
    st1.subheader("Seasonal Decompose (Additive)")
    #st1.line_chart(df[ss_col])
    st1.write("Seasonal Chart")
    st1.line_chart(decompose_result.seasonal)
    st1.write("Trend Chart")
    st1.line_chart(decompose_result.trend)
    #st2.divider()
    st2.subheader("Seasonal Decompose (Multiplicative)")
    st2.write("Seasonal Chart")
    st2.line_chart(decompose_result_mul.seasonal)
    st2.write("Trend Chart")
    st2.line_chart(decompose_result_mul.trend)
    # selectbox(col, default = 'RATE')
    # slider for seasonality period (2,3,4,5,6), default = 3
    # display decomposition chart (additive & multiplicative)
with tab4:  
    st.subheader("Model Fitting")
    mf_col = st.selectbox("Choose variable to apply model fitting:",options=var)
    train = df[mf_col]
    mf1, mf2,mf3 = st.columns(3)

    # arima
    mf1.subheader("ARIMA parameters")
    p = mf1.number_input('Enter "p":',min_value=0,max_value=3,step=1,value=1)
    d = mf1.number_input('Enter "d":',min_value=0,max_value=3,step=1,value=1)
    q = mf1.number_input('Enter "q":',min_value=0,max_value=3,step=1,value=1)
    mf1.divider()
    #mf1.divider()
    mf1.subheader("")

    arima_model = ARIMA(train, order=(p,d,q))
    arima_fit = arima_model.fit()
    mf1.pyplot(model(train,arima_fit,mf_col,'ARIMA'))
    mf1.dataframe(eval(train,arima_fit.predict()),hide_index=True,use_container_width=True)
    
    # sarima
    mf2.subheader("SARIMA parameters")
    P = mf2.number_input('Enter "P":',min_value=0,max_value=3,step=1)
    D = mf2.number_input('Enter "D":',min_value=0,max_value=3,step=1,value=1)
    Q = mf2.number_input('Enter "Q":',min_value=0,max_value=3,step=1,value=2)
    S = mf2.number_input('Enter "S":',min_value=2,max_value=6,step=1,value=6)

    sarima_model = SARIMAX(train, order=(p,d,q),seasonal_order=(P,D,Q,S))
    sarima_fit = sarima_model.fit()
    # mf2.subheader("")
    # mf2.subheader("")
    # mf2.subheader("")
    #mf2.write("")
    mf2.pyplot(model(train,sarima_fit,mf_col,'SARIMA'))
    mf2.dataframe(eval(train,sarima_fit.predict()),hide_index=True,use_container_width=True)
    # hwes
    mf3.subheader("HWES parameter")
    tr = mf3.selectbox("Select trend type:",options=['add','mul'])
    ss = mf3.selectbox("Select seasonal type:",options=['add','mul'])
    period = mf3.number_input('Enter seasonal peiod:',min_value=2,max_value=6,step=1,value=3)
    mf3.divider()
    mf3.subheader("")
    # mf3.divider()
    hwes_model = ExponentialSmoothing(train,trend=tr,seasonal=ss,seasonal_periods=period)
    hwes_fit = hwes_model.fit()
    mf3.pyplot(model(train,hwes_fit,mf_col,'HWES'))
    mf3.dataframe(eval(train,hwes_fit.fittedvalues),hide_index=True,use_container_width=True)

    # selectbox(col, default = 'RATE')
    # select p, d, q for arima [col1] default (0,1,1)
    # display eval (MAE, MAPE, RMSE)
    # select p,d ,q, P,D,Q,S for sarima [col2] default (0,1,2,6)
    # display eval (MAE, MAPE, RMSE)
    # select trend, season, period for hwes [col3] default(add,add,3)
    # display eval (MAE, MAPE, RMSE)

#1st differencing
#fig =gen_heatmap(df)
#st.pyplot(fig,clear_figure=True,use_container_width=True)