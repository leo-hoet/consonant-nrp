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
        x_nec_old, x_pos_old = self.accessor.get_xs(old_x)
        y_nec_old, y_pos_old = self.accessor.get_ys(old_x)

        for alpha in self._params.AC:
            for i, j in self._params.prereq:
                xi = self.accessor.x_val_nec(x=x, req_name=i, alpha=alpha)
                xj = self.accessor.x_val_nec(x=x, req_name=j, alpha=alpha)
                xi_old = self.accessor.x_val_nec(x=x_nec_old, req_name=i, alpha=alpha)
                xj_old = self.accessor.x_val_nec(x=x_nec_old, req_name=j, alpha=alpha)

                if xj > xi:
                    x_nec = self.accessor.x_mutate(x_nec, i, alpha, 1)

                xi = self.accessor.x_val_pos(x=x, req_name=i, alpha=alpha)
                xj = self.accessor.x_val_pos(x=x, req_name=j, alpha=alpha)
                if xj > xi:
                    x_pos = self.accessor.x_mutate(x_pos, i, alpha, 1)
        return np.concatenate((x_nec, x_pos, y_nec, y_pos))

    def changes_idx(self, x1, x2) -> List[int]:
        res = []
        for (i, (x, y)) in enumerate(zip(x1, x2)):
            if x != y:
                res.append(i)
        return res

    def _repair_alphas(self, old_x, x):
        reqs_names = self._params.effort_req.keys()

        x_nec, x_pos = self.accessor.get_xs(x)
        y_nec, y_pos = self.accessor.get_ys(x)
        x_nec_old, x_pos_old = self.accessor.get_xs(old_x)
        y_nec_old, y_pos_old = self.accessor.get_ys(old_x)

        changes = self.changes_idx(x_nec_old, x_nec)

        for change_idx in changes:
            if x_nec[change_idx] > x_nec_old[change_idx]:
                # from 0 to 1
                pass
            else:
                pass

        for alpha in self._params.AC:
            for alpha2 in [a for a in self._params.AC if a > alpha]:
                for req_name in reqs_names:
                    x_alpha_nec = self.accessor.x_val(x_nec, req_name, alpha)
                    x_alpha2_nec = self.accessor.x_val(x_nec, req_name, alpha2)
                    x_alpha_nec_old = self.accessor.x_val(x_nec_old, req_name, alpha)

                    if x_alpha_nec > x_alpha_nec_old:
                        # x_alpha_nec from 0 to 1
                        if x_alpha_nec > x_alpha2_nec:
                            x_nec = self.accessor.x_mutate(x_nec, req_name, alpha2, 1)

                        x_alpha_pos = self.accessor.x_val(x_pos, req_name, alpha)
                        x_alpha2_pos = self.accessor.x_val(x_pos, req_name, alpha2)

                        if x_alpha_pos > x_alpha2_pos:
                            x_pos = self.accessor.x_mutate(x_pos, req_name, alpha2, 1)

        for alpha in self._params.AC[::-1]:
            alphas_2 = [a for a in self._params.AC[::-1] if a < alpha]
            for alpha2 in alphas_2:
                for req_name in reqs_names:
                    x_alpha_nec = self.accessor.x_val(x_nec, req_name, alpha)
                    x_alpha2_nec = self.accessor.x_val(x_nec, req_name, alpha2)
                    x_alpha_nec_old = self.accessor.x_val(x_nec_old, req_name, alpha)

                    if x_alpha_nec_old > x_alpha_nec:
                        # x_alpha_nec from 1 to 0
                        if x_alpha_nec < x_alpha2_nec:
                            x_nec = self.accessor.x_mutate(x_nec, req_name, alpha2, 0)

                        x_alpha_pos = self.accessor.x_val(x_pos, req_name, alpha)
                        x_alpha2_pos = self.accessor.x_val(x_pos, req_name, alpha2)

                        if x_alpha_pos < x_alpha2_pos:
                            x_pos = self.accessor.x_mutate(x_pos, req_name, alpha2, 0)

        new_x = np.concatenate((x_nec, x_pos, y_nec, y_pos))
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

    def _repair(self, old_x, x):
        # x = self._repair_precedence(old_x, x)
        dbg = self._dbg_x(x)
        x = self._repair_alphas(old_x, x)
        # x = self._repair_nec_pos(x)
        return x

    def repair(self, old_X, X):
        xs = []
        for old_x, x in zip(old_X, X):
            xs.append(self._repair(old_x, x))
        res = np.row_stack(xs)
        return res
