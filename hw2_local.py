
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path="CreditCard.csv"):

    df = pd.read_csv(path)  # read the csv file into a pandas dataframe

    # turn all string columns into numeric 0/1 values so we can do math
    x = pd.DataFrame({
        "Gender": (df["Gender"] == "M").astype(int),       # male becomes 1, female becomes 0
        "CarOwner": (df["CarOwner"] == "Y").astype(int),   # yes becomes 1, no becomes 0
        "PropertyOwner": (df["PropertyOwner"] == "Y").astype(int), # same encoding for property
        "Children": df["#Children"].astype(float),         # number of children already numeric
        "WorkPhone": df["WorkPhone"].astype(int),          # keep work phone as 0 or 1
        "Email": df["Email_ID"].astype(int),               # keep email as 0 or 1
    }).to_numpy(dtype=float)  # finally turn the dataframe into a numpy array

    # the output label is whether credit was approved (1) or denied (0)
    y = df["CreditApprove"].astype(float).to_numpy()
    return x, y  # return features and labels

def error(w, x, y):
    """compute mean squared error for a weight vector w"""
    f = x @ w  # prediction = dot product of inputs and weights
    return np.mean((f - y) ** 2)  # average squared difference from the real answer

def neighbor_oneflip(w):
    """
    make a list of all neighbors of w that differ by one entry
    example: [1,1,1] -> [[-1,1,1], [1,-1,1], [1,1,-1]]
    """
    nbs = []  # will store the neighbors
    for j in range(len(w)):   # go through each position in w
        v = w.copy()          # make a copy of w
        v[j] = -v[j]          # flip the sign at position j
        nbs.append(v)         # add this new vector to the neighbor list
    return nbs

def hill_climb(x, y, max_rounds=1000, seed=None):
    rng = np.random.default_rng(seed)  # random generator, controlled by seed
    w = rng.choice([-1, 1], size=x.shape[1])  # pick a random starting weight vector
    best_err = error(w, x, y)  # calculate error at the start
    history = [best_err]       # keep track of error values as we search

    # try improving up to max_rounds times
    for _ in range(max_rounds):
        improved = False  # flag to check if we find any improvement this round
        for i in range(len(w)):   # look at each position to flip
            neighbor = w.copy()   # make a neighbor by flipping one position
            neighbor[i] *= -1     # flip sign at index i
            current_err = error(neighbor, x, y)  # calculate error of this neighbor

            if current_err < best_err:  # if this neighbor is better
                w = neighbor            # move to this neighbor
                best_err = current_err  # update best error
                history.append(best_err)  # save this new error in history
                improved = True         # mark that we improved this round
        if not improved:  # if no neighbor was better, we are stuck in a local minimum
            break
    return w, np.array(history)  # return the best weights and the error history

def multi_start_hill_climb(x, y, runs=20, max_rounds=1000):
    """
    run hill climbing several times with different random starts
    return the solution with the longest history (most improvements)
    """
    best_w, best_history = None, []  # placeholders for best result
    for seed in range(runs):  # try different random seeds
        w, history = hill_climb(x, y, max_rounds=max_rounds, seed=seed)
        if len(history) > len(best_history):  # keep the run that had more improvement steps
            best_w, best_history = w, history
    return best_w, np.array(best_history)

if __name__ == "__main__":
    x, y = load_data()  # load dataset into features and labels

    # run hill climbing many times and pick the best improvement path
    w_best, errs = multi_start_hill_climb(x, y, runs=50, max_rounds=1000)

    # print the final best weight vector and its error
    print("Optimal w (local search):", w_best)
    print("Optimal er(w):", errs[-1])

    # plot the error values over rounds to see convergence
    plt.figure()
    plt.plot(range(len(errs)), errs, marker="o")
    plt.xlabel("Round of search")
    plt.ylabel("er(w)")
    plt.title("Hill Climbing Convergence (Local Search)")
    plt.tight_layout()
    plt.savefig("figure2_local.png", dpi=200)  # save figure as image file
    plt.show()  # also display the figure
