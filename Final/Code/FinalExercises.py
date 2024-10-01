import numpy as np
import math
import scipy.stats as stats
from scipy.integrate import quad

def problemOne():
    def euroOptions():
        s0 = 100
        sigma = 0.2
        r = 0.04
        K = 100
        n = 3
        T = 1
        dt = T / n

        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r * dt) - d) / (u - d)
        p_down = 1 - p

        stock_price_tree = []
        put_payoff_tree = [0] * ((n + 1) * (n + 2) // 2 + 1)
        call_payoff_tree = [0] * ((n + 1) * (n + 2) // 2 + 1)

        # Building the stock price tree
        for j in range(n + 1):
            level = []
            for i in range(j + 1):
                stock_price = s0 * (u ** (j - i)) * (d ** i)
                level.append(stock_price)
            stock_price_tree.append(level)

        flattened_stock_price_tree = [item for sublist in stock_price_tree for item in sublist]

        # Printing the stock price tree
        print("Stock Price Tree:")
        for j, level in enumerate(stock_price_tree):
            print(f"\t Level {j}: ", end="")
            print(" ".join([f"{price:.2f}" for price in level]))

        leaf_nodes_start_index = n * (n + 1) // 2
        num_nodes = (n + 1) * (n + 2) // 2

        # Calculate payoff at leaf nodes
        for j in range(leaf_nodes_start_index + 1, num_nodes + 1):
            put_payoff_tree[j] = max(K - flattened_stock_price_tree[j - 1], 0)

        # Calculate the option price at each node by working backwards
        current_index = leaf_nodes_start_index
        num_time_frame = n
        next_bottom_corner = current_index

        for j in range(leaf_nodes_start_index, 0, -1):
            put_payoff_tree[j] = np.exp(-r * dt) * (
                        put_payoff_tree[j + num_time_frame] * p + put_payoff_tree[j + num_time_frame + 1] * (1 - p))
            current_index -= 1
            if current_index == (next_bottom_corner - num_time_frame):
                next_bottom_corner = current_index
                num_time_frame -= 1

        # Function to print the put price tree
        def print_price_tree(payoff_tree, n):
            index = 0
            for j in range(n + 1):
                print(f"\t Level {j}: ", end="")
                for i in range(j + 1):
                    index += 1
                    print(f"{payoff_tree[index]:.2f}", end=" ")
                print()

        # Print the put price tree
        print("\nEuropean Put Price Tree:")
        print_price_tree(put_payoff_tree, n)

        # Calculate payoff at leaf nodes
        for j in range(leaf_nodes_start_index + 1, num_nodes + 1):
            call_payoff_tree[j] = max(flattened_stock_price_tree[j - 1] - K, 0)

        # Calculate the option price at each node by working backwards
        current_index = leaf_nodes_start_index
        num_time_frame = n
        next_bottom_corner = current_index

        for j in range(leaf_nodes_start_index, 0, -1):
            call_payoff_tree[j] = np.exp(-r * dt) * (
                        call_payoff_tree[j + num_time_frame] * p + call_payoff_tree[j + num_time_frame + 1] * (1 - p))
            current_index -= 1
            if current_index == (next_bottom_corner - num_time_frame):
                next_bottom_corner = current_index
                num_time_frame -= 1

        # Print the put price tree
        print("\nEuropean Call Price Tree:")
        print_price_tree(call_payoff_tree, n)


    def ameriOptions():
        s0 = 100
        sigma = 0.2
        r = 0.04
        K = 100
        n = 3
        T = 1
        dt = T / n

        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r * dt) - d) / (u - d)
        p_down = 1 - p

        stock_price_tree = []
        put_payoff_tree = [0] * ((n + 1) * (n + 2) // 2)
        call_payoff_tree = [0] * ((n + 1) * (n + 2) // 2)

        # Building the stock price tree
        for j in range(n + 1):
            level = []
            for i in range(j + 1):
                stock_price = s0 * (u ** (j - i)) * (d ** i)
                level.append(stock_price)
            stock_price_tree.append(level)

        flattened_stock_price_tree = [item for sublist in stock_price_tree for item in sublist]

        leaf_nodes_start_index = n * (n + 1) // 2
        num_nodes = (n + 1) * (n + 2) // 2

        # Calculate payoff at leaf nodes
        for j in range(leaf_nodes_start_index, num_nodes):
            put_payoff_tree[j] = max(K - flattened_stock_price_tree[j], 0)

        # Calculate the option price at each node by working backwards
        current_index = leaf_nodes_start_index
        num_time_frame = n
        next_bottom_corner = current_index

        for j in range(leaf_nodes_start_index - 1, -1, -1):
            hold_value = np.exp(-r * dt) * (
                        put_payoff_tree[j + num_time_frame] * p + put_payoff_tree[j + num_time_frame + 1] * (1 - p))
            exercise_value = max(0, K - flattened_stock_price_tree[j])
            put_payoff_tree[j] = max(hold_value, exercise_value)
            current_index -= 1
            if current_index == (next_bottom_corner - num_time_frame):
                next_bottom_corner = current_index
                num_time_frame -= 1

        # Function to print the put price tree
        def print_price_tree(payoff_tree, n):
            index = 0
            for j in range(n + 1):
                print(f"\tLevel {j}: ", end="")
                for i in range(j + 1):
                    print(f"{payoff_tree[index]:.2f}", end=" ")
                    index += 1
                print()

        # Print the put price tree
        print("\nAmerican Put Price Tree:")
        print_price_tree(put_payoff_tree, n)

        # Calculate payoff at leaf nodes
        for j in range(leaf_nodes_start_index, num_nodes):
            call_payoff_tree[j] = max(flattened_stock_price_tree[j] - K, 0)

        # Calculate the option price at each node by working backwards
        current_index = leaf_nodes_start_index
        num_time_frame = n
        next_bottom_corner = current_index

        for j in range(leaf_nodes_start_index - 1, -1, -1):
            hold_value_call = np.exp(-r * dt) * (
                        call_payoff_tree[j + num_time_frame] * p + call_payoff_tree[j + num_time_frame + 1] * (1 - p))
            exercise_value_call = max(0, flattened_stock_price_tree[j] - K)
            call_payoff_tree[j] = max(hold_value_call, exercise_value_call)
            current_index -= 1
            if current_index == (next_bottom_corner - num_time_frame):
                next_bottom_corner = current_index
                num_time_frame -= 1

        # Print the put price tree
        print("\nAmerican Call Price Tree:")
        print_price_tree(call_payoff_tree, n)


    def binaryOptions():
        s0 = 100
        sigma = 0.2
        r = 0.04
        K = 100
        n = 3
        T = 1
        dt = T / n
        payoff = 100

        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r * dt) - d) / (u - d)
        p_down = 1 - p

        stock_price_tree = []
        payoff_tree = [0] * ((n + 1) * (n + 2) // 2 )
        ameri_payoff_tree = [0] * ((n + 1) * (n + 2) // 2)
        leaf_nodes_start_index = (n * (n + 1) // 2)
        num_nodes = (n + 1) * (n + 2) // 2

        # Building the stock price tree
        for j in range(1, n+2):
            if j==1:
              stock_price_tree.append(s0)
            k=j
            for i in range(j + 1):
                stock_price = s0 * (u ** (k)) * (d ** (j-k))
                stock_price_tree.append(stock_price)
                k-=1

        flattened_stock_price_tree = stock_price_tree


        # Calculate payoff at leaf nodes
        for j in range(leaf_nodes_start_index-1, num_nodes):
            if flattened_stock_price_tree[j] > 100:
                payoff_tree[j] = payoff
                ameri_payoff_tree[j] = payoff
            else:
                payoff_tree[j] = 0
                ameri_payoff_tree[j] = 0

        # Calculate the option price at each node by working backwards



        current_index = leaf_nodes_start_index #start in the lower right hand corner and work our way upwards from right to left.
        num_time_frame = n
        next_bottom_corner = current_index

        def payoffFunc(index):
          if round(flattened_stock_price_tree[index], 2) > 100:
            return 100
          else:
            return 0

        # calculate the regular nodes
        for j in range(leaf_nodes_start_index-1, -1, -1):
            payoff_tree[j] = np.exp(-1*r*dt)*(payoff_tree[(j+num_time_frame)]*p + payoff_tree[(j+num_time_frame+1)]*(1-p))
            current_index -= 1
            ameri_payoff_tree[j] = max(np.exp(-1*r*dt)*(ameri_payoff_tree[(j+num_time_frame)]*p + ameri_payoff_tree[(j+num_time_frame+1)]*(1-p)), payoffFunc(j))
            if current_index == (next_bottom_corner-num_time_frame):
              next_bottom_corner = current_index
              num_time_frame -= 1


        # Function to print the put price tree
        def print_price_tree(payoff_tree, n):
            index = 0
            for j in range(n + 1):
                print(f"\tLevel {j}: ", end="")
                for i in range(j + 1):
                    print(f"{payoff_tree[index]:.2f}", end=" ")
                    index += 1
                print()

        # Print the put price tree
        print("\nEuropean Binary Price Tree:")
        print_price_tree(payoff_tree, n)
        print("\nAmerican Binary Price Tree:")
        print_price_tree(ameri_payoff_tree, n)


    def quadOptions():
        s0 = 100
        sigma = 0.2
        r = 0.04
        K = 100
        n = 3
        T = 1
        dt = T / n
        payoff = 100

        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r * dt) - d) / (u - d)
        p_down = 1 - p

        stock_price_tree = []
        payoff_tree = [0] * ((n + 1) * (n + 2) // 2)
        ameri_payoff_tree = [0] * ((n + 1) * (n + 2) // 2)

        # Building the stock price tree
        for j in range(n + 1):
            level = []
            for i in range(j + 1):
                stock_price = s0 * (u ** (j - i)) * (d ** i)
                level.append(stock_price)
            stock_price_tree.append(level)

        flattened_stock_price_tree = [item for sublist in stock_price_tree for item in sublist]

        leaf_nodes_start_index = n * (n + 1) // 2
        num_nodes = (n + 1) * (n + 2) // 2

        # Calculate payoff at leaf nodes
        for j in range(leaf_nodes_start_index, num_nodes):
            if flattened_stock_price_tree[j] < 200:
                payoff_tree[j] = ((flattened_stock_price_tree[j] - 100) ** 2) / 100
                ameri_payoff_tree[j] = ((flattened_stock_price_tree[j] - 100) ** 2) / 100
            else:
                payoff_tree[j] = 0
                ameri_payoff_tree[j] = 0

        # Calculate the option price at each node by working backwards
        current_index = leaf_nodes_start_index
        num_time_frame = n
        next_bottom_corner = current_index

        for j in range(leaf_nodes_start_index - 1, -1, -1):
            payoff_tree[j] = np.exp(-r * dt) * (payoff_tree[j + num_time_frame] * p + payoff_tree[j + num_time_frame + 1] * (1 - p))
            exercise_value = 0
            if flattened_stock_price_tree[j] < 200:
                exercise_value = ((flattened_stock_price_tree[j] - 100) ** 2) / 100
            ameri_payoff_tree[j] = max(payoff_tree[j], exercise_value)
            current_index -= 1
            if current_index == (next_bottom_corner - num_time_frame):
                next_bottom_corner = current_index
                num_time_frame -= 1

        # Function to print the price tree
        def print_price_tree(payoff_tree, n):
            index = 0
            for j in range(n + 1):
                print(f"\tLevel {j}: ", end="")
                for i in range(j + 1):
                    print(f"{payoff_tree[index]:.2f}", end=" ")
                    index += 1
                print()

        # Print the put price tree
        print("\nEuropean Quadratic Price Tree:")
        print_price_tree(payoff_tree, n)
        print("\nAmerican Quadratic Price Tree:")
        print_price_tree(ameri_payoff_tree, n)


    def expOptions():
        s0 = 100
        sigma = 0.2
        r = 0.04
        K = 100
        n = 3
        T = 1
        dt = T / n
        payoff = 100

        u = np.exp(sigma * np.sqrt(dt))
        d = np.exp(-sigma * np.sqrt(dt))
        p = (np.exp(r * dt) - d) / (u - d)
        p_down = 1 - p

        stock_price_tree = []
        payoff_tree = [0] * ((n + 1) * (n + 2) // 2)
        ameri_payoff_tree = [0] * ((n + 1) * (n + 2) // 2)

        # Building the stock price tree
        for j in range(n + 1):
            level = []
            for i in range(j + 1):
                stock_price = s0 * (u ** (j - i)) * (d ** i)
                level.append(stock_price)
            stock_price_tree.append(level)

        flattened_stock_price_tree = [item for sublist in stock_price_tree for item in sublist]

        leaf_nodes_start_index = n * (n + 1) // 2
        num_nodes = (n + 1) * (n + 2) // 2

        # Calculate payoff at leaf nodes
        for j in range(leaf_nodes_start_index, num_nodes):
            if flattened_stock_price_tree[j] > 0:
                payoff_tree[j] = 100 * np.exp(-(flattened_stock_price_tree[j] / 100))
                ameri_payoff_tree[j] = 100 * np.exp(-(flattened_stock_price_tree[j] / 100))
            else:
                payoff_tree[j] = 0
                ameri_payoff_tree[j] = 0

        # Calculate the option price at each node by working backwards
        current_index = leaf_nodes_start_index
        num_time_frame = n
        next_bottom_corner = current_index

        for j in range(leaf_nodes_start_index - 1, -1, -1):
            payoff_tree[j] = np.exp(-r * dt) * (
                        payoff_tree[j + num_time_frame] * p + payoff_tree[j + num_time_frame + 1] * (1 - p))
            exercise_value = 100 * np.exp(-(flattened_stock_price_tree[j] / 100))
            ameri_payoff_tree[j] = max(payoff_tree[j], exercise_value)
            current_index -= 1
            if current_index == (next_bottom_corner - num_time_frame):
                next_bottom_corner = current_index
                num_time_frame -= 1

        # Function to print the put price tree
        def print_price_tree(payoff_tree, n):
            index = 0
            for j in range(n + 1):
                print(f"\tLevel {j}: ", end="")
                for i in range(j + 1):
                    print(f"{payoff_tree[index]:.2f}", end=" ")
                    index += 1
                print()

        # Print the put price tree
        print("\nEuropean Exponential Price Tree:")
        print_price_tree(payoff_tree, n)
        print("\nAmerican Exponential Price Tree:")
        print_price_tree(ameri_payoff_tree, n)

    print("-------------------------\nProblem 1:")
    euroOptions()
    ameriOptions()
    binaryOptions()
    expOptions()
    quadOptions()


def problemTwo():
    import numpy as np
    import math
    import scipy.stats as stats
    from scipy.integrate import quad

    def binomialEval():

        def euroOptions():
            s0 = 100
            sigma = 0.2
            r = 0.04
            K = 100
            n = 100
            T = 1
            dt = T / n

            u = np.exp(sigma * np.sqrt(dt))
            d = np.exp(-sigma * np.sqrt(dt))
            p = (np.exp(r * dt) - d) / (u - d)
            p_down = 1 - p

            stock_price_tree = []
            call_payoff_tree = [0] * ((n + 1) * (n + 2) // 2 + 1)
            sqrt_euro_payoff_tree = [0] * ((n + 1) * (n + 2) // 2 + 1)

            # Building the stock price tree
            for j in range(n + 1):
                level = []
                for i in range(j + 1):
                    stock_price = s0 * (u ** (j - i)) * (d ** i)
                    level.append(stock_price)
                stock_price_tree.append(level)

            flattened_stock_price_tree = [item for sublist in stock_price_tree for item in sublist]

            leaf_nodes_start_index = n * (n + 1) // 2
            num_nodes = (n + 1) * (n + 2) // 2

            def euroSqrtReturns(index):
                if round(flattened_stock_price_tree[index - 1], 2) > 100:
                    return 5 * math.sqrt(flattened_stock_price_tree[j - 1] - 100)
                else:
                    return 0

            # Calculate payoff at leaf nodes
            for j in range(leaf_nodes_start_index + 1, num_nodes + 1):
                call_payoff_tree[j] = max(flattened_stock_price_tree[j - 1] - K, 0)
                sqrt_euro_payoff_tree[j] = euroSqrtReturns(j)

            # Calculate the option price at each node by working backwards
            current_index = leaf_nodes_start_index
            num_time_frame = n
            next_bottom_corner = current_index

            for j in range(leaf_nodes_start_index, 0, -1):
                call_payoff_tree[j] = np.exp(-r * dt) * (
                        call_payoff_tree[j + num_time_frame] * p + call_payoff_tree[j + num_time_frame + 1] * (1 - p))
                sqrt_euro_payoff_tree[j] = np.exp(-r * dt) * (
                        sqrt_euro_payoff_tree[j + num_time_frame] * p + sqrt_euro_payoff_tree[
                    j + num_time_frame + 1] * (1 - p))
                current_index -= 1
                if current_index == (next_bottom_corner - num_time_frame):
                    next_bottom_corner = current_index
                    num_time_frame -= 1

            # Print the put price tree
            print("\nBinomial European Call Price Tree:", call_payoff_tree[1])
            print("Binomial European Sqrt Root Price: ", sqrt_euro_payoff_tree[1])

        def ameriOptions():
            s0 = 100
            sigma = 0.2
            r = 0.04
            K = 100
            n = 100
            T = 1
            dt = T / n

            u = np.exp(sigma * np.sqrt(dt))
            d = np.exp(-sigma * np.sqrt(dt))
            p = (np.exp(r * dt) - d) / (u - d)
            p_down = 1 - p

            stock_price_tree = []
            sqrt_payoff_tree_ameri = [0] * ((n + 1) * (n + 2) // 2)
            call_payoff_tree = [0] * ((n + 1) * (n + 2) // 2)

            # Building the stock price tree
            for j in range(n + 1):
                level = []
                for i in range(j + 1):
                    stock_price = s0 * (u ** (j - i)) * (d ** i)
                    level.append(stock_price)
                stock_price_tree.append(level)

            flattened_stock_price_tree = [item for sublist in stock_price_tree for item in sublist]

            leaf_nodes_start_index = n * (n + 1) // 2
            num_nodes = (n + 1) * (n + 2) // 2

            # Calculate the option price at each node by working backwards
            current_index = leaf_nodes_start_index
            num_time_frame = n
            next_bottom_corner = current_index

            def sqrtPayoffAmeriReturns(index):
                if round(flattened_stock_price_tree[index], 2) > 100:
                    return 5 * math.sqrt(flattened_stock_price_tree[j] - 100)
                else:
                    return 0

            # Calculate payoff at leaf nodes
            for j in range(leaf_nodes_start_index, num_nodes):
                call_payoff_tree[j] = max(flattened_stock_price_tree[j] - K, 0)
                sqrt_payoff_tree_ameri[j] = sqrtPayoffAmeriReturns(j)

            def payoffFunc(index):
                if round(flattened_stock_price_tree[index], 2) > 100:
                    return 5 * math.sqrt(flattened_stock_price_tree[index] - 100)
                else:
                    return 0

            for j in range(leaf_nodes_start_index - 1, -1, -1):
                hold_value_call = np.exp(-r * dt) * (
                        call_payoff_tree[j + num_time_frame] * p + call_payoff_tree[j + num_time_frame + 1] * (1 - p))
                exercise_value_call = max(0, flattened_stock_price_tree[j] - K)
                call_payoff_tree[j] = max(hold_value_call, exercise_value_call)
                ameri_hold_value_call = np.exp(-r * dt) * (
                        sqrt_payoff_tree_ameri[j + num_time_frame] * p + sqrt_payoff_tree_ameri[
                    j + num_time_frame + 1] * (1 - p))
                # ameri_exercise_value_call = max(0, flattened_stock_price_tree[j] - K)
                sqrt_payoff_tree_ameri[j] = max(ameri_hold_value_call, payoffFunc(j))
                current_index -= 1
                if current_index == (next_bottom_corner - num_time_frame):
                    next_bottom_corner = current_index
                    num_time_frame -= 1

            # Print the put price tree
            print("\nBinomial American Call Price Tree:", call_payoff_tree[0])
            print("Binomial American Sqrt Root Price: ", sqrt_payoff_tree_ameri[0])

        ameriOptions()
        euroOptions()

    def monteCarloEval():
        s0 = 100
        sigma = 0.2
        r = 0.04
        K = 100
        n = 100
        T = 1
        dt = T / n

        simuls = 1000000

        ST = np.zeros(simuls)

        for i in range(simuls):
            vec = np.random.normal(0, 1, n)
            changevec = 1 + r * dt + sigma * math.sqrt(dt) * vec
            Svals = s0 * np.cumprod(changevec)
            ST[i] = Svals[-1]

        K = 100

        def sqrtPayoffsReturn(ST):
            new_vals = np.zeros(simuls)
            for i in range(simuls):
                if round(ST[i], 2) > 100:
                    new_vals[i] = 5 * math.sqrt((ST[i] - 100))
                else:
                    new_vals[i] = 0
            return new_vals

        call_payoffs = np.maximum(ST - K, 0)
        call_price = np.exp(-r * T) * np.mean(call_payoffs)
        # sqrt_payoffs = np.where(ST > 100, 5 * np.sqrt(ST - 100), 0)
        sqrt_payoffs = sqrtPayoffsReturn(ST)
        sqrt_price = np.exp(-r * T) * np.mean(sqrt_payoffs)
        std_error = np.std(call_payoffs) / np.sqrt(simuls)
        conf_interval = stats.norm.interval(0.95, loc=call_price, scale=std_error)
        sqrt_conf_interval = stats.norm.interval(0.95, loc=sqrt_price, scale=std_error)

        # Print results
        print(f"\nMonte Carlo Call Option Price: {call_price}")
        print(f"\t95% Confidence Interval: {conf_interval}")
        print(f"\nMonte Carlo Square Root Option Price: {sqrt_price}")
        print(f"\t95% Confidence Interval: {sqrt_conf_interval}")

    def expectedPayoffEval():
        S0 = 100
        sigma = 0.2
        r = 0.04
        T = 1
        K = 100

        # Define the PDF function
        def pdf(S_T, S0, sigma, r, T):
            return (1 / (sigma * S_T * math.sqrt(2 * math.pi * T))) * math.exp(
                -((math.log(S_T / S0) - (r - 0.5 * sigma ** 2) * T) ** 2) / (2 * sigma ** 2 * T))

        def callIntegrand(S_T, S0, sigma, r, T, K):
            return (S_T - K) * pdf(S_T, S0, sigma, r, T)

        def sqrtIntegrand(S_T, S0, sigma, r, T):
            if S_T > 100:
                return (5 * math.sqrt(S_T - 100)) * pdf(S_T, S0, sigma, r, T)
            else:
                return 0

        # Perform the numerical integration
        part1 = math.exp(-r * T)

        call_integral_result, _ = quad(callIntegrand, K, np.inf, args=(S0, sigma, r, T, K))

        sqrt_integral_result = quad(sqrtIntegrand, S0, np.inf, args=(S0, sigma, r, T))[0]

        CallPrice = part1 * call_integral_result
        sqrtPrice = part1 * sqrt_integral_result
        print("\nThe Expected Payoff call price is: $", CallPrice)
        print("The Expexted Payoff square root price is: $", sqrtPrice)

    print("-------------------------\nProblem 2:")
    binomialEval()
    monteCarloEval()
    expectedPayoffEval()


problemOne()
problemTwo()