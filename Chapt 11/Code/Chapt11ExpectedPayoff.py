import math
import numpy as np
from scipy.integrate import quad

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
print("The price is: $", Price)
print("The call price is: $", CallPrice)
print("The binary price is: $", binaryPrice)
print("The quadratic price is: $", quadPrice)
print("The exponential price is: $", expPrice)
