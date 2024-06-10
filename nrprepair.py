from typing import List

import numpy as np

from nrp_consonant_helpers import ConsonantNRPParameters
from nrpaccessors import NrpAccessors


class NrpRepair:
    def __init__(self, params: ConsonantNRPParameters):
        self._params = params
        self.accessor = NrpAccessors(params)

    def _repair_precedence(self, old_x, x):
        x_nec, x_pos = self.accessor.get_xs(x)
        y_nec, y_pos = self.accessor.get_ys(x)

        for alpha in self._params.AC:
            for i, j in self._params.prereq:
                xi = self.accessor.x_val_nec(x=x, req_name=i, alpha=alpha)
                xj = self.accessor.x_val_nec(x=x, req_name=j, alpha=alpha)
                xi_old = self.accessor.x_val_nec(x=old_x, req_name=i, alpha=alpha)
                xj_old = self.accessor.x_val_nec(x=old_x, req_name=j, alpha=alpha)

                if (xi_old, xj_old) == (0, 0) and (xi, xj) == (1, 0):
                    x_nec = self.accessor.x_mutate(x_nec, j, alpha, 1)

                if (xi_old, xj_old) == (1, 1) and (xi, xj) == (0, 1):
                    x_nec = self.accessor.x_mutate(x_nec, j, alpha, 0)

                xi = self.accessor.x_val_pos(x=x, req_name=i, alpha=alpha)
                xj = self.accessor.x_val_pos(x=x, req_name=j, alpha=alpha)
                xi_old = self.accessor.x_val_nec(x=old_x, req_name=i, alpha=alpha)
                xj_old = self.accessor.x_val_nec(x=old_x, req_name=j, alpha=alpha)
                if (xi_old, xj_old) == (0, 0) and (xi, xj) == (1, 0):
                    x_pos = self.accessor.x_mutate(x_pos, j, alpha, 1)

                if (xi_old, xj_old) == (1, 1) and (xi, xj) == (0, 1):
                    x_pos = self.accessor.x_mutate(x_pos, j, alpha, 0)

        return np.concatenate((x_nec, x_pos, y_nec, y_pos))

    def _repair_matrix(self, x_nec, x_nec_old):
        # This method has some problems like if the array is in an inconsistent state it will not repair it but
        # it's good enough
        x_nec_matrix = self._dbg_x(x_nec)
        x_old_nec_matrix = self._dbg_x(x_nec_old)

        result = x_nec_matrix.copy()

        rows, cols = x_nec_matrix.shape
        for i in range(rows):
            for j in range(cols):
                x_old = x_old_nec_matrix[i, j]
                x_new = x_nec_matrix[i, j]
                if x_new == x_old:
                    continue
                # from 0 to 1
                if x_new > x_old:
                    for k in range(i, rows):
                        result[k, j] = True
                else:
                    for k in reversed(range(i)):
                        result[k, j] = False

        return result

    def _repair_alphas(self, old_x, x):
        reqs_names = self._params.effort_req.keys()

        x_nec, x_pos = self.accessor.get_xs(x)
        y_nec, y_pos = self.accessor.get_ys(x)
        x_nec_old, x_pos_old = self.accessor.get_xs(old_x)

        repaired_nec = self._repair_matrix(x_nec, x_nec_old)
        repaired_pos = self._repair_matrix(x_pos, x_pos_old)

        new_x = np.concatenate((repaired_nec.flatten(), repaired_pos.flatten(), y_nec, y_pos))
        dbg = self._dbg_x(new_x)
        return new_x

    def _dbg_x(self, X):
        reqs_names = self._params.effort_req.keys()
        res = []
        for alpha in self._params.AC:
            v = []
            for req_name in reqs_names:
                v.append(self.accessor.x_val_nec(X, req_name, alpha))
            res.append(v)
        return np.array(res, dtype=bool)

    def _repair_nec_pos(self, old_x, x):
        xs_nec, xs_pos = self.accessor.get_xs(x)
        xs_nec_old, xs_pos_old = self.accessor.get_xs(old_x)
        y_nec, y_pos = self.accessor.get_ys(x)

        for alpha in self._params.AC:
            for req in self._params.effort_req.keys():
                x_nec = self.accessor.x_val_nec(x=x, req_name=req, alpha=alpha)
                x_pos = self.accessor.x_val_pos(x=x, req_name=req, alpha=alpha)
                x_nec_old = self.accessor.x_val_nec(x=old_x, req_name=req, alpha=alpha)
                x_pos_old = self.accessor.x_val_pos(x=old_x, req_name=req, alpha=alpha)

                if (x_nec, x_pos) == (0, 0) and (x_nec_old, x_pos_old) == (1, 0):
                    xs_pos = self.accessor.x_mutate(xs_pos, req_name=req, alpha=alpha, new_val=1)

                if (x_nec, x_pos) == (1, 1) and (x_nec_old, x_pos_old) == (0, 1):
                    xs_nec = self.accessor.x_mutate(xs_nec, req_name=req, alpha=alpha, new_val=0)

        return np.concatenate((xs_nec, xs_pos, y_nec, y_pos))

    def _repair(self, old_x, x):
        x = self._repair_nec_pos(old_x, x)
        # x = self._repair_precedence(old_x, x)
        x = self._repair_alphas(old_x, x)
        return x

    def repair(self, old_X, X):
        xs = []
        for old_x, x in zip(old_X, X):
            xs.append(self._repair(old_x, x))
        res = np.row_stack(xs)
        return res
