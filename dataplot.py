import numpy as np
import matplotlib.pyplot as plt


def plot_cost(costs):
    cost_length = len(costs)
    plt.plot(list(range(cost_length)), costs)
    plt.xlabel("No. of iterations")
    plt.ylabel("Cost")
    plt.show()


