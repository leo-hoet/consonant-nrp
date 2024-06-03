from pymoo.core.mutation import Mutation
import numpy as np

from nrp_consonant_helpers import ConsonantNRPParameters
from nrprepair import NrpRepair


class NrpMutation(Mutation):

    def __init__(self, params: ConsonantNRPParameters, mut_prob=0.8):
        super().__init__()
        self._params = params
        self.mut_prob = mut_prob

    def _repair(self, X):
        return NrpRepair(params=self._params).repair(X)

    def _mutate(self, problem, X):
        self.prob_var = 0.1
        prob_var = self.get_prob_var(problem, size=(len(X), 1))
        Xp = np.copy(X)
        flip = np.random.random(X.shape) < prob_var
        Xp[flip] = ~X[flip]
        return Xp

    def _do(self, problem, X, **kwargs):
        # By inspecting X, it always contains 10 individuals of the population. I haven't found a way to change this
        X = self._mutate(problem, X)
        X = self._repair(X)
        X = np.array(X, dtype=bool)
        return X
