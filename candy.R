###############################################################
#                                                             #
# MTU 5781 Time Series Analysis                               #
# Spring 2021, Final Project                                  #
# Lorenzo Gordon & Kasia Krueger                              #
#                                                             #
###############################################################

###############################################################
#                                                             #
# Monthly U.S. Candy Production 1972-2017                     #
# https://www.kaggle.com/rtatman/us-candy-production-by-month #
#                                                             #
###############################################################

library(TSA)
library(forecast)
library(tseries)
library(stats)
library(rmarkdown)


# Import the data file
candy = read.csv(file.choose())

# Save for R-makedown
save(candy, file="candy.RData")

# Convert data to time series
candyTS = ts(round(candy$IPG3113N, digits=1), freq=12, start=c(1972,1))


# 1. Plot the data. From 7.1 How to construct a time series model

# Plot candyTS (first four years to see seasonality)
months = c("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")
plot(candyTS, type="l", xlim=c(1972,1976), ylim=c(50,150), xlab="Year",
     ylab="Production", main="U.S. Monthly Candy Production Seasonality")
points(candyTS, xlim=c(1972,1976), pch = months, col = 1, cex=0.5)

# Plot candyTS (1972-2017)
plot(candyTS, type="l", ylim=c(50,150), xlab="Year",
     ylab="Production", main="U.S. Monthly Candy Production")

# Add moving average to plot
candyTrend = ma(candyTS, order = 12)
plot(candyTS, type="l", ylim=c(50,150), xlab="Year",
     ylab="Production", main="U.S. Monthly Candy Production")
lines(candyTrend)

# decompose candy time series
candyTScomponents <- decompose(candyTS)
candyTScomponents
plot(candyTScomponents)
plot(candyTScomponents$trend, ylab="Production", main="Trend for Candy Production")
plot(candyTScomponents$seasonal, ylab="Production", ylim=c(-20, 30), main="Seasonality for Candy Production")
plot(candyTScomponents$random, ylab="Production", ylim=c(-20, 20), main="Random component of Candy Production")
plot(candyTScomponents$trend+candyTScomponents$seasonal+candyTScomponents$random,
     ylab="Production", main="Candy Trend + Seasonal + Random")

#
# data appears to have both a seasonal and overall trend
#

# Remove the trend from original data for later testing
candyTSDT = candyTScomponents$seasonal + candyTScomponents$random
candyTSDT = na.omit(candyTSDT)
plot(candyTSDT, type='l', ylab="Production", main="Detrended Candy Production")

# Plot seasonal difference
candyTS %>% diff(lag=12) %>% ggtsdisplay()

# acf functoin
Acf(candyTSDT, lag.max=100, plot=TRUE)
Pacf(candyTSDT, lag.max=100, plot=TRUE)

# adf test Augmented Dickey Fuller Test, p-value < 0.05 indicates the TS is stationary
adf.test(candyTSDT)
# p < 0.01, accept Ha residuals are stationary

# QQ plot of data
qqnorm(candyTSDT, main="QQ plot of Detrended Candy Data")
qqline(candyTSDT)
# detrended candyTS looks non-normal


# look at the first difference of the detrended data
plot(diff(candyTSDT), type='l', ylab="Production", main="First Difference Detrended Candy Production")
Acf(diff(candyTSDT), lag.max=20, plot=TRUE)
adf.test(diff(candyTSDT))
qqnorm(diff(candyTSDT), main="QQ Plot of First Difference Detrended Candy Data")
qqline(diff(candyTSDT))
# still non-normal

# Plot monthly means
candym = matrix(candyTS, ncol=12, byrow=TRUE)
month.means = apply(candym, 2, mean)
plot(month.means, type="l", main="Monthly Means of Candy Production", xlab="Month", ylab="Mean")
points(month.means, pch = months, col = 1, cex=0.9)

# Fit linear model to trend line
candyTrendLM = lm(candyTScomponents$trend ~ time(candyTScomponents$trend))
summary(candyTrendLM)

# plot candy trend with linear model (may need for forecasting)
plot(candyTScomponents$trend, ylab="Production", main="Trend for Candy Production")
abline(candyTrendLM)


# Deterministic models

# Seasonal Means model on detrended data
plot(candyTSDT, type="l", xlim=c(1972,2017), ylim=c(-30,40), xlab="Year",
     ylab="Production", main="Detrended Candy Production")

month = season(candyTSDT)
candySMwoi =  lm(candyTSDT ~ month - 1)	# fit without intercept
summary(candySMwoi)
# R^2 = .89

## compare the result with intercept
candySMwi =  lm(candyTSDT ~ month)	# fit with intercept
summary(candySMwi)

# 2. Diagnostics of the Residuals

# Residual plot (zero mean and homoskedasticity)
candySM_rid = rstandard(candySMwoi)  #stardardized residual
plot(candySM_rid, ylab="Residuals", xlab="Time",
     type="l",main="Residuals from Seasonal Means Model Detrended")
abline(h=0,lty=2)
# residuals look mean zero, some concern with variance

# Histogram of residuals
hist(candySM_rid, main="Normalized Residuals from Seasonal Model Detrended")
# looks normal, some outliers in the tails

# QQ plot of residuals
qqnorm(candySM_rid, main="QQ plot residuals from Seasonal Model Detrended")
qqline(candySM_rid)
# small problem in the lower tail

# Shapiro-Wilk test for normality
shapiro.test(candySM_rid)
# p = 0.2281, normal

# Runs test
runs(candySM_rid)
# pval=1.52e-11, observed=191, expected=268, reject independence

# Sample ACF plot
Acf(candySM_rid, main="ACF plot from residuals Seasonal Means Detrended")

# adf test Augmented Dickey Fuller Test, p-value < 0.05 indicates the TS is stationary
adf.test(candySM_rid)
# p < 0.01, accept Ha residuals are stationary



# Cosine model of detrended data
har = harmonic(candyTSDT, 1)
candyCS = lm(candyTSDT ~ har)
summary(candyCS)
# R^2 = .78

plot(ts(fitted(candyCS), freq=12, start=c(1972,1)), type="l", ylim=c(-20,20), xlab="Year",
     ylab="Production", main="Cosine Model Detrended Candy Production")

# 2. Diagnostics of the Residuals

# Residual plot (zero mean and homoskedasticity)
candyCS_rid = rstandard(candyCS)  #stardardized residual
plot(candyCS_rid, ylab="Standardized Residuals from Candy Detrended", xlab="Time",
     type="l",main="Standardized Residuals Cosine Model Detrended")
abline(h=0,lty=2)
# mean zero, but problem with variance

# Histogram of residuals
hist(candyCS_rid, main="Normalized Residuals from Cosine Model Detrended")
# looks normal, some outliers in the lower tail

# QQ plot of residuals
qqnorm(candyCS_rid, main="QQ Plot residuals from Cosine Model Detrended")
qqline(candyCS_rid)
# small problem in the upper tail, normal

# Shapiro-Wilk test for normality
shapiro.test(candyCS_rid)
# p = 0.2954, normal

# Runs test
runs(candyCS_rid)
# pvalue=5.7e-08, observed=206, expected=268, reject independence

# Residuals ACF plot
Acf(candyCS_rid, main="ACF plot of residuals from Cosine Detrended")
# looks like time series, dependent data

# Compare the fitted trends with Seasnal Means model and Cosine Trends
plot(candyTSDT, type="l", xlim=c(1972,1976), ylim=c(-20,40), xlab="Year",
     ylab="Production", main="Seasional Model VS Cosine Model")
candySM_fit = ts(fitted(candySMwoi), freq=12, start=c(1972.5,1))
lines(candySM_fit, lty=2, col="red")
candyCS_fit = ts(fitted(candyCS), freq=12, start=c(1972.5,1))
lines(candyCS_fit, lty=3, col="blue")
legend(1972, 40, legend=c("Data", "Seasonal Means", "Cosine"),
       col=c("black", "red", "blue"), lty=1:2, cex=0.8)



# Find SARIMA model stochastic model


# 4. Detrend the data
# see line 62


# 5. Choose the order of integration 'd'

# Plot a section of the raw data
months = c("J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D")
plot(candyTS, type="l", xlim=c(1972,1978), xlab="Year",
     ylab="Production", main="Monthly Candy Production")
points(candyTS, xlim=c(1972,1982), pch = months, col = 1, cex=0.8)

# plot detrended data
plot(candyTSDT, type='o', ylab="Production", main="Detrended Candy Production")

# adf test Augmented Dickey Fuller Test, p-value < 0.05 indicates the TS is stationary
adf.test(candyTSDT)
# p < 0.01, accept Ha data are stationary

# Phillips–Perron Unit Root Test
pp.test(candyTSDT)
# p < .01, no unit root

# Kpss test
kpss.test(candyTSDT)
# p > 0.1, cannot reject H0 that data is stationary

#
# Select d=0
# undifferenced data passed the adf and pp test.


# see https://otexts.com/fpp2/seasonal-arima.html
# Check auto.arima on detrended data
# Warning, this function takes several minutes
candyARIMA = auto.arima(candyTSDT)
summary(candyARIMA)
# auto.arima returned a multiplicative seasonal SARIMA model
# ARIMA(0,0,2)(1,1,2)[12] with drift
# AIC=2738.47   AICc=2738.69   BIC=2768.3

# ARIMA residuals analysis for default (0,0,2)(1,1,2)
candyARIMA = arima(candyTSDT, order=c(0,0,2), seasonal=list(order=c(1,1,2), period=12))
summary(candyARIMA)
# AIC=2736.49

plot.ts(candyARIMA$residuals, ylab="Residual", main="ARIMA (0,0,2)(1,1,2) Residuals")

Acf(candyARIMA$residuals, lag.max=24, main="ACF of the ARIMA Model Residuals")
Pacf(candyARIMA$residuals, lag.max=24, plot=TRUE)
signif(acf(candyARIMA$residuals, plot="F")$acf[1:12],2)

hist(rstandard(candyARIMA), main="Normalized Residuals from ARIMA(0,0,2)(1,1,2)")
plot(rstandard(candyARIMA))
residuals<- rstandard(candyARIMA)
plot(residuals, main="ARIMA(0,0,1)(1,1,2)[12] Residuals", ylab='Residuals')
# normal with outliers

qqnorm(candyARIMA$residuals, main="Q-Q Plot of residuals ARIMA(0,0,2)(1,1,2)")
qqline(candyARIMA$residuals)
# problem in the tails

# Shapiro-Wilk test for normality, p-value > .05 = normal
shapiro.test(candyARIMA$residuals)
# p-value = 0.08, normal

# Runs test
runs(candyARIMA$residuals)
# pvalue=0.0233, observed=242, expected=268

# adf test, p-value < 0.05 indicates the TS is stationary
adf.test(candyARIMA$residuals)
# pvalue < 0.01, accept Ha residuals are stationary

summary(candyARIMA)
confint(candyARIMA)

#
# try all ARIMA(p,d,q)(P,D,Q) for p,d,q,P,D,Q in 0:2
# take a coffee break!
minAIC = 9999999
for(p in 0:2) {
   for(d in 0:2) {
      for(q in 0:2) {
         for(P in 0:2) {
            for(D in 0:2) {
               for(Q in 0:2) {
                  tryCatch( {model = arima(x=candyTSDT, order=c(p, d, q),
                     seasonal=list(order=c(P,D,Q),period=12))},
                     error=function(e){}, warning=function(w){})
                  print(paste("(", p, ",", d, ",", q,
                     ")(", P, ",", D, ",", Q, ")", model$aic))
                  if (!is.null(model$aic) && model$aic < minAIC) {
                     minAIC = model$aic
                     minAICp = p
                     minAICd = d
                     minAICq = q
                     minAICP = P
                     minAICD = D
                     minAICQ = Q
                  }
               }
            }
         }
      }
   }
}
print(paste("Min AIC of ", minAIC, "at (", minAICp, ",", minAICd, ",", minAICq,
            ")(", minAICP, ",", minAICD, ",", minAICQ, ")"))

# order=(2,0,2), seasonal=(0,1,2) has lowest AIC
# AIC = 2629.62

# ARIMA residuals analysis for default (2,0,2)(0,1,2)
candyARIMA = arima(candyTSDT, order=c(2,0,2), seasonal=list(order=c(0,1,2), period=12))
summary(candyARIMA)
# AIC=2629.62
# note: Arima and arima differ

plot.ts(candyARIMA$residuals, ylab="Residual", main="ARIMA (2,0,2)(0,1,2) Residuals")
#Box.test(candyARIMA$residuals,lag=20, type="Ljung-Box") # H0 Model fits

checkresiduals(candyARIMA)
autoplot(candyARIMA)

Acf(candyARIMA$residuals, lag.max=24, main="ACF of the ARIMA Model Residuals")
Pacf(candyARIMA$residuals, lag.max=24, plot=TRUE)
signif(acf(candyARIMA$residuals, plot="F")$acf[1:12],2)

hist(rstandard(candyARIMA), main="Normalized Residuals from ARIMA(2,0,2)(0,1,2)")
# normal with outlier in lower tail

qqnorm(rstandard(candyARIMA), main="Q-Q Plot of residuals ARIMA(2,0,2)(0,1,2)")
qqline(rstandard(candyARIMA))
# small problem in lower tail

# Shapiro-Wilk test for normality, p-value > .05 = normal
shapiro.test(candyARIMA$residuals)
# p-value = 0.5149 normal

# Runs test
runs(candyARIMA$residuals)
# p-value = 0.697, independent

# adf test, p-value < 0.05 indicates the TS is stationary
adf.test(candyARIMA$residuals)
# P-value < 0.01, accept Ha residuals are stationary

# Box-Pierce test
Box.test(candyARIMA$residuals, lag = 6, type = "Box-Pierce", fitdf = 0)
# Cannot reject H0 residuals do not show lack of fit

# Ljung-Box test
Box.test(candyARIMA$residuals,lag=6, type="Ljung-Box") # H0 Model fits
# Cannot reject H0 residuals are independent

tsdiag(candyARIMA, gof.lag=6)

# 24-month forecast
candyARIMA = Arima(candyTSDT, order=c(2,0,2), seasonal=list(order=c(0,1,2), period=12))
plot(forecast(candyARIMA, level=.95), 24, shaded=F, col="black")
legend(2015.5, -10, legend=c("Forecast", "95% Confidence"),
       col=c("blue", "black"), lty=1:2, cex=0.8)

# Overfitting
candyARIMA = arima(candyTSDT, order=c(3,0,2), seasonal=list(order=c(0,1,2), period=12))
summary(candyARIMA)
candyARIMA = arima(candyTSDT, order=c(2,0,3), seasonal=list(order=c(0,1,2), period=12))
summary(candyARIMA)
# both overfittings have a larger AIC than the selected model

# Add the removed trend back to model

# Plot original time series vs fitted model
candyARIMA = Arima(candyTSDT, order=c(2,0,2), seasonal=list(order=c(0,1,2), period=12))
plot(candyTS,col="red", xlim=c(1972,2017), xlab="Date", ylab="Production",
     main="Original Data vs Fitted SARIMA Model")
lines(fitted(candyARIMA)+candyTScomponents$trend,col="blue", xlim=c(1972, 2017))
legend(1995.5, 80, legend=c("Original Time Series", "SARIMA Model"),
       col=c("Red", "Blue"), lty=1:2, cex=0.8)

# Plot section of original time series vs fitted model
plot(candyTS,col="red", xlim=c(2000,2010), ylim=c(70,140), xlab="Date", ylab="Production",
     main="10 Yr Original Data vs Fitted SARIMA Model")
lines(fitted(candyARIMA)+candyTScomponents$trend,col="blue", xlim=c(2000, 2010))
legend(2000.5, 90, legend=c("Original Time Series", "SARIMA Model"),
       col=c("Red", "Blue"), lty=1:2, cex=0.8)

