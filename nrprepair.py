import numpy as np

from nrp_consonant_helpers import ConsonantNRPParameters
from nrpaccessors import NrpAccessors


class NrpRepair:
    def __init__(self, params: ConsonantNRPParameters):
        self._params = params
        self.accessor = NrpAccessors(params)

    def _repair_precedence(self, x):
        for alpha in self._params.AC:
            for i, j in self._params.prereq:
                xi = self.accessor.x_val(xs=x, req_name=i, alpha=alpha)
                xj = self.accessor.x_val(xs=x, req_name=j, alpha=alpha)
                if xi >= xj:
                    continue
                x = self.accessor.x_mutate(x, i, alpha, 1)
        return x

    def _repair_alphas(self, x):
        reqs_names = self._params.effort_req.keys()

        for alpha in self._params.AC:
            for alpha2 in [a for a in self._params.AC if a > alpha]:
                for req_name in reqs_names:
                    x_alpha = self.accessor.x_val(x, req_name, alpha)
                    x_alpha2 = self.accessor.x_val(x, req_name, alpha2)
                    if x_alpha <= x_alpha2:
                        continue
                    x = self.accessor.x_mutate(x, req_name, alpha2, 1)
        return x

    def repair(self, x):
        x = self._repair_precedence(x)
        x = self._repair_alphas(x)
        return x
