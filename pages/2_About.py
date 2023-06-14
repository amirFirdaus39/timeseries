import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt

dfmw = pd.read_excel('D:/Result.xlsx',sheet_name='MOS_WEEK')
dftw = pd.read_excel('D:/Result.xlsx',sheet_name='TOTAL_WEEK')
dfrw = pd.read_excel('D:/Result.xlsx',sheet_name='RATE_WEEK')
dfmm = pd.read_excel('D:/Result.xlsx',sheet_name='RATE_MONTH')

st.title(":mag: About Project")
tab1, tab2, tab3, tab4 = st.tabs(["Time Series", "Results","Dictionary","References"])

with tab1:
    st.header("📝 Time Series Model")
    st.subheader(":interrobang: What is time series?")
    st.markdown("A time-series data is a series of data points or observations recorded at different or regular time intervals. In general, a time series is a sequence of data points taken at equally spaced time intervals. The frequency of recorded data points may be hourly, daily, weekly, monthly, quarterly or annually.")
    st.divider()

    st.subheader(":ballot_box_with_check: Components of a Time Series")
    st.markdown("- **Trend** - The trend shows a general direction of the time series data over a long period of time. A trend can be increasing(upward), decreasing(downward), or horizontal(stationary).")
    st.markdown("- **Seasonality** - A seasonality is observed when there is a distinct repeated pattern observed between regular intervals due to seasonal factors. It could be because of the month of the year, the day of the month, weekdays or even time of the day.")
    st.divider()

    st.subheader(":ballot_box_with_check: Time Series Terminology")
    st.markdown("- **Stationarity** - It shows the mean value of the series that remains constant over the time period. If past effects accumulate and the values increase towards infinity then stationarity is not met. A stationary series is one where the values of the series is not a function of time. So, the values are independent of time.")
    st.markdown("- **Differencing** - Differencing is used to make the series stationary and to control the auto-correlations. There may be some cases in time series analyses where we do not require differencing and over-differenced series can produce wrong estimates.")
    st.markdown("- **Exponential Smoothing** - Exponential smoothing in time series analysis predicts the one next period value based on the past and current value. It involves averaging of data such that the non-systematic components of each individual case or observation cancel out each other. The exponential smoothing method is used to predict the short term prediction.")
    st.divider()

    st.subheader(":white_check_mark: Data Used")
    st.markdown("Data is taken from One Flow database/dashboard that consists data from 2022 until as of May 2023. There are 2 sets of data to be used for forecasting time series; data by week and data by month. Usually One Flow is interpreted by monthly analysis but due to lack of data points, the data is aggregrated by week to increase size of data. Though data by month is also used to see if smaller data can even forecast at the same accuracy or better than its by-week counterpart. If similar result acquired for both of this data, data by-month will be used due to more simpler process (predicting 'Rate' directly instead of predicting 'MOS Completed' and 'Total MOS' and then calculate the 'Rate' in by-week data) as an equivalent trade-off.")
    st.divider()

    st.subheader(":computer: Model Used")
    st.markdown("**1) HWES - Holt-Winter’s Exponential Smoothing**")
    st.markdown("This model takes into account the trend and seasonality while doing the forecasting. This method has 3 major aspects for performing the predictions. It has an average value with the trend and seasonality.")
    st.markdown("- **Exponential Smoothing:** Simple exponential smoothing as the name suggest is used for forecasting when the data set has no trends or seasonality.")
    st.markdown("- **Holt’s Smoothing method:** Holt’s smoothing technique, also known as linear exponential smoothing, is a widely known smoothing model for forecasting data that has a trend.")
    st.markdown("- **Winter’s Smoothing method:** Winter’s smoothing technique allows us to include seasonality while making the prediction along with the trend")

    st.markdown("**2) ARIMA - AutoRegressive Integrated Moving Average**")
    st.markdown("ARIMA stands for Autoregressive Integrated Moving Average Model. It belongs to a class of models that explains a given time series based on its own past values -i.e.- its own lags and the lagged forecast errors. The equation can be used to forecast future values. Any ‘non-seasonal’ time series that exhibits patterns and is not a random white noise can be modeled with ARIMA models.The ARIMA algorithm is made of the following components:")
    st.markdown("- The **AR** stands for Auto Regression which is denoted as P, the value of P determines how the data is regressed on its past values.")
    st.markdown("- The **I** stands for Integrated or the differencing component which is denoted as d, the value of d determines the degree of difference used to make the series stationary.")
    st.markdown("- The **MA** stands for Moving Average which is denoted as q, the values of q determines the outcome of the model depends linearly on the past observations and the same goes for the errors in forecasting as they also vary linearly.")
    st.markdown("**3) SARIMA - Seasonal ARIMA**")
    st.markdown("The plain ARIMA model has a problem. It does not support seasonality. If the time series has defined seasonality, then we should go for Seasonal ARIMA model (in short SARIMA) which uses seasonal differencing. Seasonal differencing is similar to regular differencing, but, instead of subtracting consecutive terms, we subtract the value from previous season.")

    st.markdown("**4) VAR - Vector AutoRegression**")
    st.markdown("Vector Autoregression (VAR) is a forecasting algorithm that can be used when two or more time series influence each other. That is, the relationship between the time series involved is bi-directional.")
    st.markdown("In the VAR model, each variable is modeled as a linear combination of past values of itself and the past values of other variables in the system. Since you have multiple time series that influence each other, it is modeled as a system of equations with one equation per variable (time series).")
    st.divider()

    st.subheader(":mag_right: Tests Used")
    st.markdown("**1) ADF Tests - Augmented Dickey-Fuller** [stationarity test]")
    st.markdown("In probability theory and statistics, a unit root is a feature of some stochastic processes (such as random walks) that can cause problems in statistical inference involving time series models. In simple terms, the unit root is non-stationary but does not always have a trend component.")
    st.markdown("- Stationary: p-value <= 0.05 & test statistic <= critical value")
    st.markdown("- Non-Stationary:  p-value > 0.05 & test statistic > critical value")

    st.markdown("**2) KPSS Tests -  Kwiatkowski-Phillips-Schmidt-Shin** [stationarity test]")
    st.markdown("- Stationary: p-value > 0.05 & test statistic > critical value")
    st.markdown("- Non-Stationary:  p-value <= 0.05 & test statistic <= critical value")

    st.markdown("The following are the possible outcomes of applying both tests:")
    st.markdown("- **Case 1:** Both tests conclude that the given series is stationary – **The series is stationary**")
    st.markdown("- **Case 2:** Both tests conclude that the given series is non-stationary – **The series is non-stationary**")
    st.markdown("- **Case 3:** ADF concludes non-stationary, and KPSS concludes stationary – **The series is trend stationary**. To make the series strictly stationary, the trend needs to be removed in this case. Then the detrended series is checked for stationarity.")
    st.markdown("- **Case 4:** ADF concludes stationary, and KPSS concludes non-stationary – **The series is difference stationary.** Differencing is to be used to make series stationary. Then the differenced series is checked for stationarity.")

    st.markdown("**3) Granger’s Causality Test** [causation test for VAR model]")
    st.markdown("- The basis behind Vector AutoRegression is that each of the time series in the system influences each other. That is, you can predict the series with past values of itself along with other series in the system. Using Granger’s Causality Test, it’s possible to test this relationship before even building the model.")
    st.markdown("**4) Cointegration Test** [relation test for VAR model]")
    st.markdown("- Cointegration test helps to establish the presence of a statistically significant connection between two or more time series.")
    st.divider()

with tab2:
    st.header("📝 Tabulation of Results")
    st.subheader(":bar_chart: Evaluation Metrics")
    st.markdown("For this time series, we are using MAE (Mean Absolute Error), RMSE (Root Mean Square Error) and MAPE (Mean Absolute Percentage Error)")
    st.markdown("1) **MAE** - The MAE is defined as the average of the absolute difference between forecasted and true values. The lower the MAE value, the better the model; a value of zero indicates that the forecast is error-free. However, because MAE does not reveal the proportional scale of the error, it can be difficult to distinguish between large and little errors. MAE might obscure issues related to low data volume. ")
    st.markdown("2) **RMSE** - MSE is defined as the average of the error squares. It is also known as the metric that evaluates the quality of a forecasting model or predictor. MSE also takes into account variance (the difference between anticipated values) and bias (the distance of predicted value from its true value). The RMSE number is in the same unit as the projected value, which is an advantage of this technique. The wider the gap between RMSE and MAE, the more erratic the error size. This statistic can mask issues with low data volume.")
    st.markdown("3) **MAPE** - MAPE is the proportion of the average absolute difference between projected and true values divided by the true value. It works better with data that is free of zeros and extreme values because of the in-denominator. The MAPE value also takes an extreme value if this value is exceedingly tiny or huge. MAPE, like MAE, understates the impact of big but rare errors caused by extreme values. Mean Square Error can be utilized to address this issue. ")
    st.markdown("Both MAE/MAPE and RMSE measures the magnitude of errors in a set of predictions. The major difference between MAE/MAPE and RMSE is the impact of the large errors. For example, if some prediction data points are large outliers errors when compared to the ground truth, those large errors will be diluted in the mean of MAE/MAPE while RMSE score will be higher because of the square operation.")
    st.divider()

    st.subheader(":white_check_mark: Model Evaluations")
    col1, col2 = st.columns(2)
    col1.write("Train/Test for MOS_COMPLETED by Week")
    col1.dataframe(dfmw,use_container_width=True)
    # col1.bar_chart(dfmw,y='Model',x='MAE')
    col2.write("Train/Test for TOTAL_MOS by Week")
    col2.dataframe(dftw,use_container_width=True)
    col1.write("Train/Test for RATE by Week")
    col1.dataframe(dfrw,use_container_width=True)
    col2.write("Train/Test for RATE by Month")
    col2.dataframe(dfmm,use_container_width=True)
    st.divider()

    #st.subheader(":white_check_mark: Choosing Best Model")
    st.markdown("")
    # chart = alt.Chart(dfmw).mark_line().encode(
    # x=alt.X('MAE'),
    # y=alt.Y('Model'),
    # color=alt.Color("Model")
    # ).properties(title="Hello World")
    # st.altair_chart(chart, use_container_width=True)


with tab3:
    st.header("📝 One Flow Dictionary")
    dict_df =  pd.read_excel('D:/One Flow Dashboard Dictionary一条流看板指标字典.xlsx',sheet_name=0)
    dict_df['No']= dict_df['No'].astype(str)
    dict_df = dict_df.set_index(['No'])
    dict_df = dict_df.loc[~((dict_df['KCP']=='Valid DU') |(dict_df['KCP']=='Exclude List')) ]
    dict_df = dict_df[['KCP','KPI','KPI_Definition']]
    st.table(dict_df)

with tab4:
    st.header("📝 References")
    st.markdown("- Arora, A. (2022, January 31). Making predictions using a very small dataset. Retrieved from https://medium.com/@amit.arora15/making-predictions-using-a-very-small-dataset-230dd579dca8")
    st.markdown("- Demir, S., Mincev, K., Kok, K., & Paterakis, N. G. (2021). Data augmentation for time series regression: Applying transformations, autoencoders and adversarial networks to electricity price forecasting. Applied Energy, 304, 117695. doi:10.1016/j.apenergy.2021.117695")
    st.markdown("- DiBattista, J. (2022, November 18). Choosing the best ML time series model for your data. Retrieved from https://towardsdatascience.com/choosing-the-best-ml-time-series-model-for-your-data-664a7062f418")
    st.markdown("- Fitting models to short time series. (2014, March 4). Retrieved from https://robjhyndman.com/hyndsight/short-time-series/")
    st.markdown("- Maheswari, J. P. (2019, April 23). Breaking the curse of small datasets in machine learning: Part 1. Retrieved from https://towardsdatascience.com/breaking-the-curse-of-small-datasets-in-machine-learning-part-1-36f28b0c044d")
    st.markdown("- Ramírez, F. G. (2022, October 20). Forecasting with machine learning models. Retrieved from https://towardsdatascience.com/forecasting-with-machine-learning-models-95a6b6579090")
    st.markdown("- Xdurana. (2018, April 19). Multivariate analysis and correlation matrix. Retrieved from https://www.kaggle.com/code/xdurana/multivariate-analysis-and-correlation-matrix")
    #st.markdown("- ")
    #st.markdown("- ")
