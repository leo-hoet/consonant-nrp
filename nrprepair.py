import numpy as np

from nrp_consonant_helpers import ConsonantNRPParameters
from nrpaccessors import NrpAccessors


class NrpRepair:
    def __init__(self, params: ConsonantNRPParameters):
        self._params = params
        self.accessor = NrpAccessors(params)

    def _repair_precedence(self, X):
        x_nec, x_pos = self.accessor.get_xs(X)
        y_nec, y_pos = self.accessor.get_ys(X)

        for alpha in self._params.AC:
            for i, j in self._params.prereq:
                xi = self.accessor.x_val_nec(x=X, req_name=i, alpha=alpha)
                xj = self.accessor.x_val_nec(x=X, req_name=j, alpha=alpha)
                if xj > xi:
                    x_nec = self.accessor.x_mutate(x_nec, i, alpha, 1)

                xi = self.accessor.x_val_pos(x=X, req_name=i, alpha=alpha)
                xj = self.accessor.x_val_pos(x=X, req_name=j, alpha=alpha)
                if xj > xi:
                    x_pos = self.accessor.x_mutate(x_pos, i, alpha, 1)
        return np.concatenate((x_nec, x_pos, y_nec, y_pos))

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
        ys_nec, ys_pos = self.accessor.get_ys(x)
        new_xs_nec = []
        new_xs_pos = []
        for x_nec, x_pos in np.column_stack((xs_nec, xs_pos)):
            if x_nec <= x_pos:
                new_xs_nec.append(x_nec)
                new_xs_pos.append(x_pos)
                continue
            new_xs_nec.append(x_nec)
            new_xs_pos.append(1)
        return np.concatenate((new_xs_nec, new_xs_pos, ys_nec, ys_pos))

    def _repair(self, x):
        x = self._repair_precedence(x)
        x = self._repair_alphas(x)
        x = self._repair_nec_pos(x)
        return x

    def repair(self, X):
        xs = []
        for x in X:
            xs.append(self._repair(x))
        res = np.row_stack(xs)
        return res
