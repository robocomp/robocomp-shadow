import nevergrad as ng

class Nevergrad_Proxy():
    def __init__(self, loss_fuction):
        self.loss_fuction = loss_fuction
        # build a parameter providing a dict value:
        self.params = ng.p.Dict(
            # logarithmically distributed float
            # log=ng.p.Log(lower=0.01, upper=1.0),
            curvature=ng.p.Scalar(init=0, lower=-30, upper=30).set_integer_casting(),
            arc=ng.p.Scalar(init=200, lower=50, upper=450).set_integer_casting(),
            lane_width=ng.p.Scalar(init=100, lower=50, upper=450).set_integer_casting(),
            projection=ng.p.Scalar(init=2, lower=0, upper=5).set_integer_casting()
        )
        self.optimizer = ng.optimizers.DiscreteDE(parametrization=self.params, budget=100, num_workers=1)
        self.optimizer = ng.optimizers.MultiDiscrete(parametrization=self.params, budget=100, num_workers=1)
        self.recommendation = self.params

    def optimize(self):
        self.optimizer.suggest(self.recommendation.value)
        for _ in range(self.optimizer.budget):
            x = self.optimizer.ask()
            loss = self.loss_fuction(x.value)
            self.optimizer.tell(x, loss)
        self.recommendation = self.optimizer.provide_recommendation()
        print("Optimum:", self.recommendation.value, "Loss:", loss)
        return self.recommendation.value