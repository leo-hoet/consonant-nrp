from pymoo.core.mutation import Mutation
import numpy as np

from nrp_consonant_helpers import ConsonantNRPParameters


class NrpMutation(Mutation):

    def __int__(self, params: ConsonantNRPParameters, mut_prob=0.1):
        super().__init__()
        self._params = params
        self.mut_prob = mut_prob

    def _repair_x(self, X):
        raise NotImplementedError()

    def _repair(self, X):
        return X

    def _mutate(self, X):
        Xp = np.copy(X)
        flip = np.random.random(X.shape) < self.mut_prob
        Xp[flip] = ~X[flip]
        return Xp

    def _do(self, problem, X, **kwargs):
        # By inspecting X, it always contains 10 individuals of the population. I haven't found a way to change this
        X_mutated = self._mutate(X)
        X_repaired = self._repair(X_mutated)
        return X_repaired
