from typing import Tuple, List

from pymoo.algorithms.soo.nonconvex import ga
from pymoo.core.problem import ElementwiseProblem
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.optimize import minimize

from nrp_consonant_helpers import nrp_example_data, ConsonantNRPParameters


class ConsonantFuzzyNRP(ElementwiseProblem):
    @property
    def len_x(self):
        return self._p.len_req * self._p.len_ac

    @property
    def len_y(self):
        return self._p.len_customers * self._p.len_ac

    def get_ys(self, x):
        y_pos = x[:, 0:((self.len_y * self._p.len_ac) - 1)]
        y_nec = x[:, ((self.len_y * self._p.len_ac) - 1):]
        return y_pos, y_nec

    def _y_val(self, y, c_name, alpha):
        customers = self._p.customers
        keys = list(customers.keys())
        customer_index = keys.index(c_name)  # 0 index
        alpha_index = self._p.AC.index(alpha)
        return y[customer_index + alpha_index * self._p.len_ac]

    def y_val_pos(self, x, customer_name: str, alpha: float):
        y_pos, _ = self.get_ys(x)
        return self._y_val(y_pos, customer_name, alpha)

    def y_val_nec(self, x, customer_name: str, alpha: float):
        _, y_nec = self.get_ys(x)
        return self._y_val(y_nec, customer_name, alpha)

    def __init__(self, params: ConsonantNRPParameters):
        self._p = params

        x_pos = np.zeros(self.len_x * len(params.AC))
        x_nec = np.zeros(self.len_x * len(params.AC))

        y_pos = np.zeros(self.len_y * len(params.AC))
        y_nec = np.zeros(self.len_y * len(params.AC))

        all_vars = np.concatenate([x_nec, x_pos, y_nec, y_pos])
        super().__init__(
            n_var=all_vars.size,
            n_obj=1,
            n_ieq_constr=1,  # TODO: this
            xl=np.zeros(all_vars.size),
            xu=np.ones(all_vars.size)
        )

    @property
    def _extended_alpha(self):
        alphas = sorted(self._p.AC)
        alphas = [0] + alphas
        extended_alpha = alphas + alphas[-1::-1]
        return extended_alpha

    @property
    def p(self):
        extended_alpha = self._extended_alpha
        p = [
            0.5 * (
                    max(extended_alpha[1:i + 1]) -
                    max(extended_alpha[0:i]) +
                    max(extended_alpha[i:-1]) -
                    max(extended_alpha[i + 1:])
            )
            for i in range(1, len(extended_alpha) - 1)
        ]
        return p

    def a(self, x):
        alphas = self._p.AC
        a = []
        for j, alpha in enumerate(alphas):
            nec_sum = sum(
                self._p.customers[customer] * self.y_val_nec(x, customer, alpha) for customer in
                self._p.customers.keys()
            )
            a.append(nec_sum)

        for j, alpha in enumerate(reversed(alphas)):
            pos_sum = sum(
                self._p.customers[customer] * self.y_val_pos(x, customer, alpha) for customer in
                self._p.customers.keys()
            )
            a.append(pos_sum)
        return a

    def get_xs(self, x):
        x_pos = x[:, 0:((self.len_x * self._p.len_ac) - 1)]
        x_nec = x[:, ((self.len_x * self._p.len_ac) - 1):]
        return x_pos, x_nec

    def _calculate_obj_function(self, x):
        p = self.p
        a = self.a(x)
        assert len(p) == len(a), "p and a does not have the same length"
        return sum(map(lambda z, y: z * y, p, a))

    # x: 1 x NVar
    def _evaluate(self, x, out, *args, **kwargs):
        res = self._calculate_obj_function(x)
        out["F"] = res
        out["G"] = 0.1 - out["F"]


def main():
    params = nrp_example_data()
    problem = ConsonantFuzzyNRP(params)
    algol = ga.GA(
        sampling=BinaryRandomSampling(),
        mutation=BitflipMutation(),
        eliminate_duplicates=True,
        pop_size=100
    )

    selection = RandomSelection()
    crossover = SBX()

    res = minimize(
        problem=problem,
        algorithm=algol,
        termination=('n_gen', 100),
        verbose=False,
        selection=selection,
        crossover=crossover
    )


if __name__ == '__main__':
    main()
