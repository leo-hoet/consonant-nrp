from nptyping import Shape, NDArray, Float
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

    def _x_val(self, xs, req_name, alpha):
        req_index = list(self._p.effort_req.keys()).index(req_name)
        alpha_index = self._p.AC.index(alpha)
        return xs[req_index + (self._p.len_ac * alpha_index)]

    def x_nec_pos(self, x, req_name: str, alpha: float):
        x_nec, _ = self.get_xs(x)
        return self._x_val(x_nec, req_name, alpha)

    def x_val_pos(self, x, req_name: str, alpha: float):
        _, x_pos = self.get_xs(x)
        return self._x_val(x_pos, req_name, alpha)

    def number_of_constraints(self) -> int:
        constraints_for_disponibility_pos = self._p.len_ac
        constraints_for_disponibility_nec = self._p.len_ac

        total = constraints_for_disponibility_pos + constraints_for_disponibility_nec
        return total

    def __init__(self, params: ConsonantNRPParameters):
        self._p = params

        x_pos = np.zeros(self.len_x)
        x_nec = np.zeros(self.len_x)

        y_pos = np.zeros(self.len_y)
        y_nec = np.zeros(self.len_y)

        all_vars = np.concatenate([x_nec, x_pos, y_nec, y_pos])
        n = self.number_of_constraints()
        super().__init__(
            n_var=all_vars.size,
            n_obj=1,
            n_ieq_constr=n,
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

    def disponibility_rule_pos(self, x) -> NDArray[Shape["self._p.len_ac"], float]:
        p1, p2, p3, p4 = self._p.p
        constraint_values = []

        for alpha in self._p.AC:
            sum1 = 0
            sum2 = 0
            for req, effort in self._p.effort_req.items():
                e1, e2, e3, e4 = effort
                x_pos = self.x_val_pos(x, req, alpha)
                sum1 += e2 * x_pos
                sum2 += (e2 - e1) * x_pos
            left_side = p3 - sum1
            right_side = (alpha - 1) * (p4 - p3 + sum2)
            # This is done bc pymoo wants constraint in the form of <= 0
            # check  https://pymoo.org/getting_started/part_2.html
            constraint_val = right_side - left_side
            constraint_values.append(constraint_val)
        assert self._p.len_ac == len(constraint_values), f"expected {self._p.len_ac} got {len(constraint_values)=} "
        return np.array(constraint_values)

    def disponibility_rule_nec(self, x) -> NDArray[Shape["self._p.len_ac"], float]:
        p1, p2, p3, p4 = self._p.p
        constraint_values = []

        for alpha in self._p.AC:
            sum1 = 0
            sum2 = 0
            for req, effort in self._p.effort_req.items():
                e1, e2, e3, e4 = effort
                sum1 += e3 * self.x_nec_pos(x, req, alpha)
                sum2 += (e4 - e3) * self.x_nec_pos(x, req, alpha)
            left_side = p2 - sum1
            right_side = (1 - alpha) * (p2 - p1 + sum2)
            constraint_val = right_side - left_side
            constraint_values.append(constraint_val)
        assert self._p.len_ac == len(constraint_values), f"expected {self._p.len_ac} got {len(constraint_values)=} "
        return np.array(constraint_values)

    def _calculate_obj_function(self, x):
        p = self.p
        a = self.a(x)
        assert len(p) == len(a), "p and a does not have the same length"
        fitness = sum(map(lambda z, y: z * y, p, a))
        fitness_to_minimize = (-1) * fitness
        return fitness_to_minimize

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
