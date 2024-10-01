# Input ticker symbols for two stocks into console.

import matplotlib.pyplot as plt
import math
import numpy as np
import yfinance as yf
import pylab
import scipy.stats as stats
import statsmodels.api as sm
from datetime import datetime, timedelta


def chosenStock(ticker):
    today = datetime.today()
    last_week = today - timedelta(days=7)
    start_date = last_week - timedelta(days=365 * 10) - timedelta(days=7)
    end_date = last_week

    history_ticker = yf.Ticker(ticker).history(period="10y")
    history_ticker_weekly = yf.Ticker(ticker).history(period="10y", interval="1wk")
    history_ticker_weekprior = yf.Ticker(ticker).history(start=start_date, end=end_date, interval="1wk")
    stockpricedisplay(ticker, history_ticker)
    logreturndisplay(ticker, history_ticker_weekly, history_ticker_weekprior)
    quantiledisplay(ticker, logreturn(history_ticker_weekly))
    randomWalk(ticker, history_ticker)


def stockpricedisplay(ticker, history_ticker):
    plt.style.use('seaborn-v0_8-darkgrid')
    history_ticker["Close"].plot(title=f"{ticker} Stock Price")
    plt.show()
    plt.clf()


def logreturndisplay(ticker, history_ticker, history_ticker_prior):
    logreturn_ticker = logreturn(history_ticker)
    logreturn_prior = logreturn(history_ticker_prior)
    plt.hist(logreturn_ticker, bins='auto')
    plt.title(f"{ticker} Log Return")
    plt.show()
    plt.clf()
    scatterdisplay(ticker, logreturn_ticker, logreturn_prior)


def logreturn(history_ticker):
    logreturn_ticker = [0]
    i = 0
    for price in history_ticker["Close"].values:
        if i > 1:
            logreturn_ticker.append(math.log(price / history_ticker["Close"].values[i - 1]))
        elif i == 1:
            logreturn_ticker[0] = (math.log(price / history_ticker["Close"].values[i - 1]))
        i += 1
    return logreturn_ticker


def quantiledisplay(ticker, history_ticker):
    stats.probplot(history_ticker, dist="norm", plot=pylab)
    pylab.title(f"{ticker} Quantile Plot")
    pylab.show()


def scatterdisplay(ticker, history_ticker, ticker_prior):
    # Convert log returns to numpy arrays and ensure they are the same length
    min_length = min(len(history_ticker), len(ticker_prior))
    x = np.array(history_ticker[:min_length])
    y = np.array(ticker_prior[:min_length])

    # Fit regression line
    X = sm.add_constant(x)  # Adds a constant term to the predictor
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    # Calculate R^2
    r_squared = model.rsquared

    # Create scatter plot
    plt.scatter(x, y, alpha=0.5)

    # Plot regression line
    plt.plot(x, predictions, color='red', linewidth=2)

    # Add equation and R^2 value
    plt.text(0.05, 0.95, f'y = {model.params[1]:.4f}x + {model.params[0]:.4f}\n$R^2$ = {r_squared:.4f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top')

    plt.xlabel('Current Log Return')
    plt.ylabel('Prior Log Return')
    plt.title(f'{ticker} Scatter Plot with Regression Line')
    plt.show()
    plt.clf()


def randomWalk(ticker, history_ticker):
    length = 2520
    S0 = history_ticker["Close"].values[-1]
    T = 10
    dt = 1 / 252
    mu = 0.10
    sigma = 0.30
    print(S0)
    simuls = 100000

    ST = np.zeros(simuls)
    sigma_hat = np.zeros(simuls)
    mu_hat = np.zeros(simuls)

    for i in range(simuls):
        vec = np.random.normal(0, 1, length)
        changevec = 1 + mu * dt + sigma * math.sqrt(dt) * vec
        Svals = S0 * np.cumprod(changevec)
        ST[i] = Svals[-1]
        mu_hat[i] = np.log(Svals[-1] / S0) / T
        logreturns = np.log(Svals[1:] / Svals[:-1])
        sigma_hat[i] = np.std(logreturns) * math.sqrt(1 / dt)

    # Print some results
    print(f"S0: {S0}")
    print(f"First 10 ST values: {ST[:10]}")
    print(f"First 10 mu_hat values: {mu_hat[:10]}")
    print(f"First 10 sigma_hat values: {sigma_hat[:10]}")
    plt.hist(mu_hat, bins='auto')
    plt.title(f"{ticker} Mu")
    plt.show()
    plt.clf()


stock_1 = input("Enter stock ticker: ")
stock_2 = input("Enter stock ticker: ")

chosenStock(stock_1)
chosenStock(stock_2)
