import numpy as np


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


euroOptions()
ameriOptions()
binaryOptions()
quadOptions()
expOptions()
