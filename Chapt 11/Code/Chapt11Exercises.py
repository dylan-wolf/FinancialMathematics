import math
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

def exerciseOne():
    # Parameters
    S0 = 100
    sigma = 0.2
    r = 0.04
    T = 1
    K = 100
    M=100

    # Define the PDF function
    def pdf(S_T, S0, sigma, r, T):
        return (1 / (sigma * S_T * math.sqrt(2 * math.pi * T))) * math.exp(
            -((math.log(S_T / S0) - (r - 0.5 * sigma ** 2) * T) ** 2) / (2 * sigma ** 2 * T))


    def putIntegrand(S_T, S0, sigma, r, T, K):
        return (K - S_T) * pdf(S_T, S0, sigma, r, T)

    def callIntegrand(S_T, S0, sigma, r, T, K):
        return (S_T-K) * pdf(S_T, S0, sigma, r, T)

    def binaryIntegrand(S_T, S0, sigma, r, T, M):
        return M * pdf(S_T, S0, sigma, r, T)

    def quadraIntegrand(S_T, S0, sigma, r, T):
        return (((S_T-100)**2)/100) * pdf(S_T, S0, sigma, r, T)

    def expIntegrand(S_T, S0, sigma, r, T):
        return (100*math.exp(-S_T/100)) * pdf(S_T, S0, sigma, r, T)

    # Perform the numerical integration
    part1 = math.exp(-r * T)
    put_integral_result, _ = quad(putIntegrand, 0, K, args=(S0, sigma, r, T, K))

    call_integral_result, _ = quad(callIntegrand, K, np.inf, args=(S0, sigma, r, T, K))

    V = quad(binaryIntegrand, M, np.inf, args=(S0, sigma, r, T, M))[0]

    quad_integral_result = quad(quadraIntegrand, 0, np.inf, args=(S0, sigma, r, T))[0]

    exp_integral_result = quad(expIntegrand, 0, np.inf, args=(S0, sigma, r, T))[0]



    # Calculate the price
    Price = part1 * put_integral_result
    CallPrice = part1 * call_integral_result
    binaryPrice = part1 * V
    quadPrice = part1 * quad_integral_result
    expPrice = part1 * exp_integral_result
    print("Exercise 1:")
    print("\tThe put price is: $", Price)
    print("\tThe call price is: $", CallPrice)
    print("\tThe binary price is: $", binaryPrice)
    print("\tThe quadratic price is: $", quadPrice)
    print("\tThe exponential price is: $", expPrice)


def exerciseTwo():
    S0 = 100
    sigma = 0.25
    r = 0.05
    shares_n = 100
    k_put = 80
    put_n=100

    T = 2
    K = 100

    def pdf(S_T, S0, sigma, r, T):
        return (1 / (sigma * S_T * math.sqrt(2 * math.pi * T))) * math.exp(
            -((math.log(S_T / S0) - (r - 0.5 * sigma ** 2) * T) ** 2) / (2 * sigma ** 2 * T))


    def putIntegrand(S_T, S0, sigma, r, T, K):
        return (K - S_T) * pdf(S_T, S0, sigma, r, T)

    def BlackScholesPut(S_T, S0, sigma, r, T, K):
      return (-S_T * norm.cdf(-d1(S_T, S0, sigma, r, T, K))) + math.exp(-r*T) * K * norm.cdf(-d2(S_T, S0, sigma, r, T, K))

    def BlackScholes(S_T, S0, sigma, r, T, K):
      return ((S_T * norm.cdf(d1(S_T, S0, sigma, r, T, K))) - math.exp(-r*T) * K * norm.cdf(d2(S_T, S0, sigma, r, T, K)))

    def findK (S_T, S0, sigma, r, T, K):
      strike = 150
      for i in range(150):
        if round(BlackScholesPut(S_T, S0, sigma, r, T, K),2) == round(BlackScholes(S_T, S0, sigma, r, T, strike),2):
          return strike
        else:
          strike = strike + 0.1
      return False
      # return S0 * norm.cdf(d1(S0, S0, K, sigma, r, T)) - K * math.exp(-r * T) * norm.cdf(d2(S0, S0, K, sigma, r, T)) - CallPrice
      # return S_T * (math.exp(2.92)-math.exp((r+1/2*sigma**2)*T)/(sigma*math.sqrt(T)))


    def d1(S_T, S0, sigma, r, T, K):
      return ((math.log(S_T/K) + (r+(1/2)*sigma ** 2)*T)/(sigma*math.sqrt(T)))

    def d2(S_T, S0, sigma, r, T, K):
      return (d1(S_T, S0, sigma, r, T, K)-sigma*math.sqrt(T))

    def callIntegrand(S_T, S0, sigma, r, T, K):
        return (S_T-K) * pdf(S_T, S0, sigma, r, T)

    def createPlot(k_put):
      x1=np.arange(0,k_put, 5)
      y1= np.full(x1.shape, -3051.71)

      x2 = np.arange(80, 164, 5)
      y2 = 100 * x2 - 206082.06

      x3 = np.arange(163, 201, 5)
      y3 = np.full(x3.shape, 5248.29)

      x_values = np.concatenate([x1, x2, x3])
      y_values = np.concatenate([y1, y2, y3])

      data = pd.DataFrame({'xValues': x_values, 'yValues': y_values})

      plt.figure(figsize=(10, 6))
      plt.axhline(y=0, color='black', linewidth=1)
      plt.plot(data['xValues'], data['yValues'], color='red')
      plt.xlabel('ST')
      plt.ylabel('Profit')
      plt.title('Profit vs. ST')
      plt.grid(True)
      plt.show()


    def call_price_from_put(S0, K, r, T, put_price):
        return put_price + S0 - K * math.exp(-r * T)


    call_strike = findK(S0, S0, sigma, r, T, k_put)


    interest = math.exp(-r * T)
    put_integral_result, _ = quad(putIntegrand, 0, k_put, args=(S0, sigma, r, T, k_put))

    call_integral_result, _ = quad(callIntegrand, call_strike, np.inf, args=(S0, sigma, r, T, call_strike))




    put_Price_BS = put_integral_result * math.exp(-r*T)

    put_Price = interest * put_integral_result
    CallPrice = interest * call_integral_result
    call_price = call_price_from_put(S0, K, r, T, put_Price)
    print("\nExercise 2:")
    print("\tThe price the put of this option via Black-Scholes is: $", BlackScholesPut(S0, S0, sigma, r, T, k_put))
    print("\tThe price the put of this option via Expected Payoff is: $", put_Price)
    print("\tThe strike price for the Call Option should be: $", call_strike)
    print("\tThe call price is: $", CallPrice)
    print("\tThe total cost for our options is: $", 200*(CallPrice - put_Price)*math.exp(r*T))
    print("\tThe total cost for our portfolio at the time of expiry is: $", 10000*math.exp(r*T)+200*(CallPrice-put_Price)*math.exp(r*T))


def exerciseThree():

    S0 = 100
    sigma = 0.25
    r = 0.03
    T = 2
    K = 200
    K_b = 50

    def pdf(S_T, S0, sigma, r, T):
        return (1 / (sigma * S_T * math.sqrt(2 * math.pi * T))) * math.exp(
            -((math.log(S_T / S0) - (r - 0.5 * sigma ** 2) * T) ** 2) / (2 * sigma ** 2 * T))


    def putIntegrand(S_T, S0, sigma, r, T, K):
        return (K - S_T) * pdf(S_T, S0, sigma, r, T)

    def BlackScholesPut(S_T, S0, sigma, r, T, K):
      return (-S_T * norm.cdf(-d1(S_T, S0, sigma, r, T, K))) + math.exp(-r*T) * K * norm.cdf(-d2(S_T, S0, sigma, r, T, K))

    def BlackScholes(S_T, S0, sigma, r, T, K):
      return ((S_T * norm.cdf(d1(S_T, S0, sigma, r, T, K))) - math.exp(-r*T) * K * norm.cdf(d2(S_T, S0, sigma, r, T, K)))

    def d1(S_T, S0, sigma, r, T, K):
      return ((math.log(S_T/K) + (r+(1/2)*sigma ** 2)*T)/(sigma*math.sqrt(T)))

    def d2(S_T, S0, sigma, r, T, K):
      return (d1(S_T, S0, sigma, r, T, K)-sigma*math.sqrt(T))

    def callIntegrand(S_T, S0, sigma, r, T, K):
        return (S_T-K) * pdf(S_T, S0, sigma, r, T)

    part1 = math.exp(-r * T)
    put_integral_result, _ = quad(putIntegrand, np.inf, K, args=(S0, sigma, r, T, K))

    call_integral_result, _ = quad(callIntegrand, K, np.inf, args=(S0, sigma, r, T, K))




    put_Price_BS = put_integral_result * math.exp(-r*T)

    put_Price = part1 * put_integral_result
    CallPrice = part1 * call_integral_result
    print("\nExercise 3:")
    print("\t3.A: $", CallPrice)
    print("\t3.B: $", BlackScholesPut(S0, S0, sigma, r, T, K_b))
    print("\tThe total cost for this is: $", (BlackScholes(S0, S0, sigma, r, T, K)+BlackScholesPut(S0, S0, sigma, r, T, K_b))*math.exp(r*T)*1000000)
    print("\tThe probability that we will make the max profit is:", norm.cdf(d2(S0, S0, sigma, r, T, K_b)) - norm.cdf(d2(S0, S0, sigma, r, T, K)))
    print("\tThe probability that the Call Option will end in the money is: ", norm.cdf(d2(S0, S0, sigma, r, T, K))*100, "%")

def exerciseFour():
    # Parameters
    S0 = 100
    T = 1
    callCost = 20
    putCost = 10
    K = 100

    # Define the PDF function
    def BlackScholes(S_T, S0, sigma, r, T, K):
      return ((S_T * norm.cdf(d1(S_T, S0, sigma, r, T, K))) - math.exp(-r*T) * K * norm.cdf(d2(S_T, S0, sigma, r, T, K)))

    def d1(S_T, S0, sigma, r, T, K):
      return ((math.log(S_T/K) + (r+(1/2)*sigma ** 2)*T)/(sigma*math.sqrt(T)))

    def d2(S_T, S0, sigma, r, T, K):
      return (d1(S_T, S0, sigma, r, T, K)-sigma*math.sqrt(T))


    r = -(math.log((S0+putCost - callCost)/K)/T)

    sigma = 0

    sigma_price = 0

    while sigma_price != callCost:
      if sigma_price > callCost:
          break
      sigma = sigma + 0.00001
      sigma_price = BlackScholes(S0, S0, sigma, r, T, K)

    print("\nExercise 4:")
    print("\tThe riskless rate is: ", r)
    print("\tThe implied volatility is: ", sigma)




exerciseOne()
exerciseTwo()
exerciseThree()
exerciseFour()