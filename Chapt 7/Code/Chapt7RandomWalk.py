import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def callOption():
    strike_k = 100
    sigma = 0.3
    s0 = 100
    T=1
    n=52
    dt= 1/n
    r=0.05

    strike_alt = 110
    knockout_barrier = 150

    simuls = 1000000

    ST = np.zeros(simuls)

    S_max = np.zeros(simuls)
    S_min = np.zeros(simuls)
    S_ave = np.zeros(simuls)
    S_lookback = np.zeros(simuls)
    S_knockIn = np.zeros(simuls)
    S_knockout = np.zeros(simuls)
    S_AsianCall = np.zeros(simuls)

    for i in range(simuls):
        vec = np.random.normal(0,1, n)
        changevec = 1 + r * dt + sigma * math.sqrt(dt) * vec
        Svals = s0 * np.cumprod(changevec)
        S_max[i] = np.max(Svals)
        S_min[i] = np.min(Svals)
        S_ave[i] = np.mean(Svals)
        S_lookback[i] = np.maximum(S_max[i]-strike_alt, 0)
        if S_max[i] < knockout_barrier:
            S_knockout[i] = np.maximum(Svals[-1]-strike_alt, 0)
        else:
            S_knockout[i] = 0
        ST[i] = Svals[-1]
        S_AsianCall[i] = np.maximum(S_ave[i]-strike_alt,0)
        if np.max(Svals) > strike_alt:
            S_knockIn[i] = np.maximum(strike_alt-ST[i], 0)
        else:
            S_knockIn[i] = 0

    knockInPutPrice = np.exp(-r * T) * np.mean(S_knockIn)
    call_payoffs = np.maximum(ST-strike_k, 0)
    call_price = np.exp(-r*T)*np.mean(call_payoffs)
    lookbackCallPrice = np.exp(-r*T)*np.mean(S_lookback)
    knockOutCallPrice = np.exp(-r*T)*np.mean(S_knockout)
    asianCallPrice = np.exp(-r*T)*np.mean(S_AsianCall)

    std_error = np.std(call_payoffs)/np.sqrt(simuls)


    plt.hist(S_max, bins=50, edgecolor='black', alpha=0.7, color='blue')
    plt.grid(visible=True, linestyle="--")
    plt.axvline(np.mean(S_max), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(S_max):.2f}')
    plt.axvline(np.mean(S_max) - np.std(S_max), color='blue', linestyle='dashed', linewidth=2,
                label=f'-1 Std Dev: {np.mean(S_max) - np.std(S_max):.2f}')
    plt.axvline(np.mean(S_max) + np.std(S_max), color='blue', linestyle='dashed', linewidth=2,
                label=f'+1 Std Dev: {np.mean(S_max) + np.std(S_max):.2f}')

    plt.title("Q1: Distribution of Max Stock Prices from Simulated Random Walks")
    plt.xlabel("Max Stock Price for Each Walk")
    plt.ylabel("Frequency")
    plt.show()
    plt.clf()

    plt.hist(S_min, bins=50, edgecolor='black', alpha=0.7, color='blue')
    plt.grid(visible=True, linestyle="--")
    plt.axvline(np.mean(S_min), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(S_min):.2f}')
    plt.axvline(np.mean(S_min) - np.std(S_min), color='blue', linestyle='dashed', linewidth=2,
                label=f'-1 Std Dev: {np.mean(S_min) - np.std(S_min):.2f}')
    plt.axvline(np.mean(S_min) + np.std(S_min), color='blue', linestyle='dashed', linewidth=2,
                label=f'+1 Std Dev: {np.mean(S_min) + np.std(S_min):.2f}')

    plt.title("Q2: Distribution of Min Stock Prices from Simulated Random Walks")
    plt.xlabel("Min Stock Price for Each Walk")
    plt.ylabel("Frequency")
    plt.show()
    plt.clf()

    plt.hist(S_ave, bins=50, edgecolor='black', alpha=0.7, color='blue')
    plt.grid(visible=True, linestyle="--")
    plt.axvline(np.mean(S_ave), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(S_ave):.2f}')
    plt.axvline(np.mean(S_ave) - np.std(S_ave), color='blue', linestyle='dashed', linewidth=2,
                label=f'-1 Std Dev: {np.mean(S_ave) - np.std(S_ave):.2f}')
    plt.axvline(np.mean(S_ave) + np.std(S_ave), color='blue', linestyle='dashed', linewidth=2,
                label=f'+1 Std Dev: {np.mean(S_ave) + np.std(S_ave):.2f}')

    plt.title("Q3: Distribution of Average Stock Prices from Simulated Random Walks")
    plt.xlabel("Average Stock Price for Each Walk")
    plt.ylabel("Frequency")
    plt.show()
    plt.clf()

    print(f"Q4: Lookback Call Price: {lookbackCallPrice}")
    print(f"Q5: Knock Out Call Price: {knockOutCallPrice}")
    print(f"Q6: Knock-In Put Price: {knockInPutPrice}")
    print(f"Q7: Asian Call Price: {asianCallPrice}")

    simuls_vol = 100000

    volatility_levels = np.arange(0, 0.525, 0.025)

    num_vol_levels = len(volatility_levels)

    ST_volatility_measure = np.zeros((simuls_vol, num_vol_levels))

    for j, vol_sigma in enumerate(volatility_levels):
        vol_payoffs = np.zeros(simuls_vol)
        for i in range(simuls_vol):
            S_vals_vol = np.zeros(n)
            S_vals_vol[0] = s0
            for t in range(1,n):
                S_vals_vol[t] = S_vals_vol[t-1] * (1 + r*dt + vol_sigma*np.sqrt(dt)*np.random.normal(0,1))

            max_Vol = np.max(S_vals_vol)
            if max_Vol < knockout_barrier:
                vol_payoffs[i] = np.maximum(S_vals_vol[-1] - strike_alt, 0)
            else:
                vol_payoffs[i] = 0
        ST_volatility_measure[:, j] = vol_payoffs

    vol_knockOutCallPrice = np.exp(-r * T) * np.mean(ST_volatility_measure, axis=0)

    plt.plot(volatility_levels, vol_knockOutCallPrice, marker='o')
    plt.xlabel('Volatility (sigma)')
    plt.ylabel('Average Final Stock Price')
    plt.title('Q8: Average Final Stock Price vs. Volatility')
    plt.grid(True)
    plt.show()


callOption()


