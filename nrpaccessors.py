from nrp_consonant_helpers import ConsonantNRPParameters
import numpy as np


class NrpAccessors:
    def __init__(self, params: ConsonantNRPParameters):
        self._p = params

    @property
    def len_x(self):
        return self._p.len_req * self._p.len_ac

    @property
    def len_y(self):
        return self._p.len_customers * self._p.len_ac

    def _y_val(self, y, c_name, alpha):
        customers = self._p.customers
        keys = list(customers.keys())
        customer_index = keys.index(str(c_name))  # 0 index
        alpha_index = self._p.AC.index(alpha)
        return y[customer_index + alpha_index * self._p.len_ac]

    def y_val_pos(self, x, customer_name: str, alpha: float):
        y_pos, _ = self.get_ys(x)
        return self._y_val(y_pos, customer_name, alpha)

    def y_val_nec(self, x, customer_name: str, alpha: float):
        _, y_nec = self.get_ys(x)
        return self._y_val(y_nec, customer_name, alpha)

    def x_mutate(self, x, req_name: str, alpha: str, new_val: float):
        x_copy = np.array(x, copy=True)

        req_index = list(self._p.effort_req.keys()).index(str(req_name))
        alpha_index = self._p.AC.index(alpha)

        x_copy[req_index + (self._p.len_ac * alpha_index)] = new_val
        return x_copy

    def x_val(self, xs, req_name, alpha):
        req_index = list(self._p.effort_req.keys()).index(str(req_name))
        alpha_index = self._p.AC.index(alpha)
        return xs[req_index + (self._p.len_ac * alpha_index)]

    def x_val_nec(self, x, req_name: str, alpha: float):
        x_nec, _ = self.get_xs(x)
        return self.x_val(x_nec, req_name, alpha)

    def x_val_pos(self, x, req_name: str, alpha: float):
        _, x_pos = self.get_xs(x)
        return self.x_val(x_pos, req_name, alpha)

    def get_ys(self, x):
        offset = 2 * self.len_x  # 2 times x, one for nec and one for pos
        y_pos = x[offset: (offset + self.len_y)]

        offset += self.len_y
        y_nec = x[offset: (offset + self.len_y)]

        offset += self.len_y
        assert offset == len(x), "Ofsset is not equal to len(x)"
        return y_pos, y_nec

    def get_xs(self, x):
        offset = 0
        x_nec = x[offset:(offset + self.len_x)]

        offset += self.len_x
        x_pos = x[offset:(offset + self.len_x)]

        return x_nec, x_pos
