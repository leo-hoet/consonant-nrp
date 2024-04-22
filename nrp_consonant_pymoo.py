from pymoo.algorithms.soo.nonconvex import ga
from pymoo.core.problem import Problem
import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.selection.rnd import RandomSelection
from pymoo.optimize import minimize

from nrp_consonant_helpers import nrp_example_data, ConsonantNRPParameters


class ConsonantFuzzyNRP(Problem):
    @property
    def len_x(self):
        return self._p.len_req * self._p.len_ac

    @property
    def len_y(self):
        return self._p.len_customers * self._p.len_ac

    def __init__(self, params: ConsonantNRPParameters):
        self._p = params

        x_pos = np.zeros(self.len_x)
        x_nec = np.zeros(self.len_x)

        y_pos = np.zeros(self.len_y)
        y_nec = np.zeros(self.len_y)

        all_vars = np.concatenate([x_nec, x_pos, y_nec, y_pos])
        super().__init__(
            n_var=all_vars.size,
            n_obj=1,
            n_ieq_constr=1,  # TODO: this
            xl=np.zeros(all_vars.size),
            xu=np.ones(all_vars.size)
        )

    def get_xs(self, x):
        x_pos = x[:, 0:self.len_x]
        x_nec = x[:, self.len_x:2 * self.len_x]
        return x_pos, x_nec

    def get_ys(self, x):
        y_pos = x[:, 0:self.len_y]
        y_nec = x[:, self.len_y:2 * self.len_y]
        return y_pos, y_nec

    def _calculate_obj_function(self, x):
        pass

    # x: PopSize x NVar
    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.sum((x - 0.5) ** 2, axis=1)
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
