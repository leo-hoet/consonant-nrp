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

    def _repair_nec_pos(self, x):
        xs_nec, xs_pos = self.accessor.get_xs(x)
        new_xs_nec = []
        new_xs_pos = []
        for x_nec, x_pos in np.column_stack((xs_nec, xs_pos)):
            if x_nec <= x_pos:
                new_xs_nec.append(x_nec)
                new_xs_pos.append(x_pos)
            new_xs_nec.append(x_nec)
            new_xs_pos.append(1)
        return np.concatenate((new_xs_nec, new_xs_pos))



    def repair(self, x):
        x = self._repair_precedence(x)
        x = self._repair_alphas(x)
        x = self._repair_nec_pos(x)
        return x
