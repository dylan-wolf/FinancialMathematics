import math
import numpy as np
import scipy.stats as stats

def main():
    callOption()
    putOption()
    binaryOption()
    quadraticOption()
    exponentialOption()

def callOption():

    sigma = 0.2
    s_0 = 100
    r = 0.04
    T = 1
    n = 52
    delta_t = 1 / n

    simuls = 1000000

    ST = np.zeros(simuls)

    for i in range(simuls):
        vec = np.random.normal(0, 1, n)
        changevec = 1 + r * delta_t + sigma * math.sqrt(delta_t) * vec
        Svals = s_0 * np.cumprod(changevec)
        ST[i] = Svals[-1]

    K = 100
    call_payoffs = np.maximum(ST - K, 0)
    call_price = np.exp(-r * T) * np.mean(call_payoffs)
    std_error = np.std(call_payoffs) / np.sqrt(simuls)
    conf_interval = stats.norm.interval(0.95, loc=call_price, scale=std_error)

    # Print results
    print(f"Call Option Price: {call_price}")
    print(f"95% Confidence Interval: {conf_interval}")

def putOption():
    sigma = 0.2
    s_0 = 100
    r = 0.04
    T = 1
    n = 52
    delta_t = 1 / n

    simuls = 1000000

    ST = np.zeros(simuls)

    for i in range(simuls):
        vec = np.random.normal(0, 1, n)
        changevec = 1 + r * delta_t + sigma * math.sqrt(delta_t) * vec
        Svals = s_0 * np.cumprod(changevec)
        ST[i] = Svals[-1]

    K = 100
    put_payoffs = np.maximum(K-ST, 0)
    put_price = np.exp(-r * T) * np.mean(put_payoffs)
    std_error = np.std(put_payoffs) / np.sqrt(simuls)
    conf_interval = stats.norm.interval(0.95, loc=put_price, scale=std_error)

    # Print results
    print(f"Put Option Price: {put_price}")
    print(f"95% Confidence Interval: {conf_interval}")

def binaryOption():
    sigma = 0.2
    s_0 = 100
    r = 0.04
    payoff = 100
    T = 1
    n = 52
    delta_t = 1 / n

    simuls = 1000000

    ST = np.zeros(simuls)

    for i in range(simuls):
        vec = np.random.normal(0, 1, n)
        changevec = 1 + r * delta_t + sigma * math.sqrt(delta_t) * vec
        Svals = s_0 * np.cumprod(changevec)
        ST[i] = Svals[-1]

    K = 100
    binary_payoffs = np.where(ST > K, payoff, 0)
    binary_price = np.exp(-r * T) * np.mean(binary_payoffs)
    std_error = np.std(binary_payoffs) / np.sqrt(simuls)
    conf_interval = stats.norm.interval(0.95, loc=binary_price, scale=std_error)

    # Print results
    print(f"Binary Option Price: {binary_price}")
    print(f"95% Confidence Interval: {conf_interval}")


def quadraticOption():
    sigma = 0.2
    s_0 = 100
    r = 0.04
    T = 1
    n = 52
    delta_t = 1 / n

    simuls = 1000000

    ST = np.zeros(simuls)

    for i in range(simuls):
        vec = np.random.normal(0, 1, n)
        changevec = 1 + r * delta_t + sigma * math.sqrt(delta_t) * vec
        Svals = s_0 * np.cumprod(changevec)
        ST[i] = Svals[-1]

    K = 100
    payoff = (ST-100)**2/100
    binary_payoffs = np.where(ST < 200, payoff, 0)
    binary_price = np.exp(-r * T) * np.mean(binary_payoffs)
    std_error = np.std(binary_payoffs) / np.sqrt(simuls)
    conf_interval = stats.norm.interval(0.95, loc=binary_price, scale=std_error)

    # Print results
    print(f"Quadratic Option Price: {binary_price}")
    print(f"95% Confidence Interval: {conf_interval}")

def exponentialOption():
    sigma = 0.2
    s_0 = 100
    r = 0.04
    T = 1
    n = 52
    delta_t = 1 / n

    simuls = 1000000

    ST = np.zeros(simuls)

    for i in range(simuls):
        vec = np.random.normal(0, 1, n)
        changevec = 1 + r * delta_t + sigma * math.sqrt(delta_t) * vec
        Svals = s_0 * np.cumprod(changevec)
        ST[i] = Svals[-1]

    K = 100
    payoff = 100*np.exp(-ST/100)
    binary_payoffs = np.where(ST > 0, payoff, 0)
    binary_price = np.exp(-r * T) * np.mean(binary_payoffs)
    std_error = np.std(binary_payoffs) / np.sqrt(simuls)
    conf_interval = stats.norm.interval(0.95, loc=binary_price, scale=std_error)

    # Print results
    print(f"Exponential Option Price: {binary_price}")
    print(f"95% Confidence Interval: {conf_interval}")


main()