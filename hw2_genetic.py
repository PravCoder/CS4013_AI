import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path="CreditCard.csv"):
    """
    Load and encode the CreditCard dataset.
    """
    df = pd.read_csv(path)  # read the dataset into a pandas DataFrame

    # convert categorical values into numbers the algorithm can use
    x = pd.DataFrame({
        "Gender": (df["Gender"] == "M").astype(int),       # M=1, F=0
        "CarOwner": (df["CarOwner"] == "Y").astype(int),   # Y=1, N=0
        "PropertyOwner": (df["PropertyOwner"] == "Y").astype(int), # Y=1, N=0
        "Children": df["#Children"].astype(float),         # already numeric
        "WorkPhone": df["WorkPhone"].astype(int),          # already 0/1
        "Email": df["Email_ID"].astype(int),               # already 0/1
    }).to_numpy(dtype=float)  # convert DataFrame to numpy array

    # target column: whether the application was approved (1) or not (0)
    y = df["CreditApprove"].astype(float).to_numpy()
    return x, y

def error(w, x, y):
    """Compute mean squared error er(w)."""
    f = x @ w  # linear combination of attributes using weights w
    return np.mean((f - y) ** 2)  # average squared difference between prediction and true label

def fitness(w, x, y):
    """Fitness = exp(-error)."""
    # lower error gives higher fitness, since exp(-small) is close to 1
    return np.exp(-error(w, x, y))

def initialize_population(pop_size, dim=6, seed=0):
    # randomly create a population of weight vectors, each entry is -1 or +1
    rng = np.random.default_rng(seed)
    return rng.choice([-1.0, 1.0], size=(pop_size, dim))

def select_parents(pop, x, y, rng):
    # calculate fitness for each candidate
    fits = np.array([fitness(w, x, y) for w in pop])
    # turn fitness into probabilities for selection
    probs = fits / (fits.sum() + 1e-12)  # add small term to avoid divide by zero
    # randomly pick 2 parents based on their fitness probabilities
    parents_idx = rng.choice(len(pop), size=2, p=probs, replace=False)
    return pop[parents_idx]

def crossover(p1, p2):
    """Midpoint crossover: first 3 from p1, last 3 from p2."""
    child1 = np.concatenate([p1[:3], p2[3:]])  # child takes left half of p1, right half of p2
    child2 = np.concatenate([p2[:3], p1[3:]])  # child takes left half of p2, right half of p1
    return child1, child2

def mutate(w, mutation_rate=0.05, rng=None):
    """Flip each bit with prob = mutation_rate."""
    if rng is None:
        rng = np.random.default_rng()
    v = w.copy()  # copy so we don’t overwrite the original
    for j in range(len(v)):
        if rng.random() < mutation_rate:  # flip with some small chance
            v[j] = -v[j]                  # flip sign (1 -> -1, -1 -> 1)
    return v

def genetic_algorithm(x, y, pop_size=20, generations=100, mutation_rate=0.05, seed=0):
    # initialize random number generator and population
    rng = np.random.default_rng(seed)
    pop = initialize_population(pop_size, dim=x.shape[1], seed=seed)

    # start with the first individual as the best
    best_w = pop[0].copy()
    best_e = error(best_w, x, y)

    # keep track of best error after each generation
    best_errors = []

    for _ in range(generations):
        # compute errors for the whole population
        errs = [error(w, x, y) for w in pop]
        gen_best_idx = np.argmin(errs)  # index of lowest error in current generation
        gen_best_e = errs[gen_best_idx]

        # update global best if we found a better solution
        if gen_best_e < best_e:
            best_e = gen_best_e
            best_w = pop[gen_best_idx].copy()

        best_errors.append(best_e)  # record the best-so-far error

        # build the next generation
        new_pop = []
        while len(new_pop) < pop_size:
            # pick two parents based on fitness
            p1, p2 = select_parents(pop, x, y, rng)
            # make two children using crossover
            c1, c2 = crossover(p1, p2)
            # randomly flip bits with small chance
            c1 = mutate(c1, mutation_rate, rng)
            c2 = mutate(c2, mutation_rate, rng)
            new_pop.extend([c1, c2])  # add both children to new population

        # ensure the population size is correct
        pop = np.array(new_pop[:pop_size])

    # return the best solution found and the list of errors per generation
    return best_w, np.array(best_errors)

if __name__ == "__main__":
    # load the dataset from file
    x, y = load_data("CreditCard.csv")

    # run the genetic algorithm
    w_ga, errs_ga = genetic_algorithm(x, y, pop_size=20, generations=100, mutation_rate=0.05, seed=0)

    # print the best solution and its error
    print("Optimal w (genetic algorithm):", w_ga)
    print("Optimal er(w):", errs_ga[-1])

    # plot the error curve over generations
    plt.figure()
    plt.plot(range(len(errs_ga)), errs_ga, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Best-so-far er(w)")
    plt.title("Genetic Algorithm Convergence")
    plt.tight_layout()
    plt.savefig("figure3_genetic.png", dpi=200)  # save figure
    plt.show()  # also display it
