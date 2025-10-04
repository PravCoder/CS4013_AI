import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path="CreditCard.csv"):

    df = pd.read_csv(path)  # read the dataset into a DataFrame

    # encode all the categorical values into numbers
    x = pd.DataFrame({
        "Gender": (df["Gender"] == "M").astype(int),       # 1 if male, 0 if female
        "CarOwner": (df["CarOwner"] == "Y").astype(int),   # 1 if owns a car, 0 otherwise
        "PropertyOwner": (df["PropertyOwner"] == "Y").astype(int), # 1 if owns property
        "Children": df["#Children"].astype(float),         # number of kids (numeric)
        "WorkPhone": df["WorkPhone"].astype(int),          # 1 if work phone, else 0
        "Email": df["Email_ID"].astype(int),               # 1 if email, else 0
    }).to_numpy(dtype=float)  # convert features to numpy array

    # target (label) is whether their credit was approved (1) or denied (0)
    y = df["CreditApprove"].astype(float).to_numpy()
    return x, y

def error(w, x, y):
    """Compute mean squared error er(w)."""
    f = x @ w  # prediction = linear combination of features and weights
    return np.mean((f - y) ** 2)  # squared difference between prediction and true value

def neighbor_oneflip(w):
    """
    Generate all neighbors of w by flipping exactly one entry.
    Example: [1,1,1] -> neighbors = [[-1,1,1], [1,-1,1], [1,1,-1]]
    """
    nbs = []  # will hold all neighbors
    for j in range(len(w)):
        v = w.copy()   # copy current weight vector
        v[j] = -v[j]   # flip the j-th element (1 -> -1 or -1 -> 1)
        nbs.append(v)  # add the new vector to neighbors list
    return nbs

def hill_climb(x, y, max_rounds=1000, seed=0):
    """
    Hill climbing local search:
    - Start from a random w ∈ {-1, +1}^6
    - Look at all neighbors (1-flip away)
    - Move to the neighbor if it reduces error
    - Stop when no better neighbor exists
    """
    rng = np.random.default_rng(seed)  # set random generator for reproducibility
    # pick a random starting point: each entry is -1 or +1
    w = rng.choice([-1.0, 1.0], size=x.shape[1])
    # keep track of errors after each round (for plotting)
    errors = [error(w, x, y)]

    # loop for a maximum number of rounds
    for _ in range(max_rounds):
        # calculate error for all neighbors
        candidates = [(error(v, x, y), v) for v in neighbor_oneflip(w)]
        # pick the neighbor with the smallest error
        best_e, best_v = min(candidates, key=lambda t: t[0])

        if best_e < errors[-1]:
            # if best neighbor is better, move there
            w = best_v
            errors.append(best_e)  # record new best error
        else:
            # if no neighbor improves, we reached a local minimum
            break

    # return the final weight vector and the history of errors
    return w, np.array(errors)

if __name__ == "__main__":
    # Step 1: load dataset
    x, y = load_data("CreditCard.csv")

    # Step 2: run hill climbing search
    w_best, errs = hill_climb(x, y, max_rounds=1000, seed=0)

    # Step 3: print the best solution found
    print("Optimal w (local search):", w_best)
    print("Optimal er(w):", errs[-1])

    # Step 4: plot the convergence (error vs. round of search)
    plt.figure()
    plt.plot(range(len(errs)), errs, marker="o")
    plt.xlabel("Round of search")      # x-axis label
    plt.ylabel("er(w)")                # y-axis label
    plt.title("Hill Climbing Convergence (Local Search)")  # plot title
    plt.tight_layout()
    plt.savefig("figure2_local.png", dpi=200)  # save figure to file
    plt.show()  # display the figure
