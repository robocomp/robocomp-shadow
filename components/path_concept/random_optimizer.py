import numpy as np
from numpy.random import default_rng

class Random_Optimizer():
    def __init__(self, loss_function, params, initial_value=[], max_iterations=100):
        self.loss_function = loss_function
        self.recommendation = initial_value
        self.max_iterations = max_iterations
        self.params = params
        self.initial_value = initial_value
        self.best_score = np.Inf
        self.rng = default_rng()

    def optimize(self):
        self.suggest(self.recommendation)
        for _ in range(self.max_iterations):
            x = self.ask()
            loss = self.loss_fuction(x)
            self.tell(x, loss)
        self.recommendation = self.provide_recommendation()
        print("Optimum:", self.recommendation, "Loss:", loss)
        return self.recommendation

    def ask(self):
        if self.suggested_values:
            self.suggested_values = False
            return self.initial_value
        # sample params
        values = []
        for i in range(len(self.params)):
            mu = self.best_score[i]
            sigma = 3
            values.append(self.rng.normal(mu, sigma, 1))

        return values

    def tell(self):
        if self.loss < self.best_score:
            self.best_score = self.loss

    def provide_recommendation(self):
        return self.best_score

    def suggest(self, values):
        self.suggested_values = True
        self.initial_value = values
