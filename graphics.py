from typing import List

import numpy as np
import matplotlib.pyplot as plt


def ga_progression(scores: List[float]):
    fig, ax = plt.subplots()
    x = range(len(scores))
    ax.scatter(x=x, y=scores, )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness")
    plt.show()


def display_pareto_matrix(problem, F):
    xl, xu = problem.bounds()
    fig, ax = plt.subplots()
    ax.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
    plt.title("Objective Space")
    ax.set_xlabel("Constraint violation")
    ax.set_ylabel("Fitness")

    plt.show()
