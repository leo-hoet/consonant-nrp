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
        customer_index = keys.index(str(c_name))  # 0 index
        alpha_index = self._p.AC.index(alpha)
        return y[customer_index + alpha_index * self._p.len_ac]

    def y_val_pos(self, x, customer_name: str, alpha: float):
        y_pos, _ = self.get_ys(x)
        return self._y_val(y_pos, customer_name, alpha)

    def y_val_nec(self, x, customer_name: str, alpha: float):
        _, y_nec = self.get_ys(x)
        return self._y_val(y_nec, customer_name, alpha)

    def _x_val(self, xs, req_name, alpha):
        req_index = list(self._p.effort_req.keys()).index(str(req_name))
        alpha_index = self._p.AC.index(alpha)
        return xs[req_index + (self._p.len_ac * alpha_index)]

    def x_val_nec(self, x, req_name: str, alpha: float):
        x_nec, _ = self.get_xs(x)
        return self._x_val(x_nec, req_name, alpha)

    def x_val_pos(self, x, req_name: str, alpha: float):
        _, x_pos = self.get_xs(x)
        return self._x_val(x_pos, req_name, alpha)

    def number_of_constraints(self) -> int:
        constraints_for_disponibility_pos = self._p.len_ac
        constraints_for_disponibility_nec = self._p.len_ac

        constraints_for_precedence_rule_nec = self._p.len_ac * len(self._p.prereq)
        constraints_for_precedence_rule_pos = self._p.len_ac * len(self._p.prereq)

        constraints_interest_pos = self._p.len_ac * len(self._p.interests)
        constraints_interest_nec = self._p.len_ac * len(self._p.interests)

        constraints_nested_plan_pos = self._p.len_ac * len(self._p.effort_req)
        constraints_nested_plan_rec = self._p.len_ac * len(self._p.effort_req)

        constraints_nec_to_pos = len(self._p.effort_req)

        total = (
                constraints_for_disponibility_pos +
                constraints_for_disponibility_nec +
                constraints_for_precedence_rule_nec +
                constraints_for_precedence_rule_pos +
                constraints_interest_pos +
                constraints_interest_nec +
                constraints_nested_plan_pos +
                constraints_nested_plan_rec +
                constraints_nec_to_pos
        )
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

    def disponibility_rule_pos(self, x):
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

    def disponibility_rule_nec(self, x):
        p1, p2, p3, p4 = self._p.p
        constraint_values = []

        for alpha in self._p.AC:
            sum1 = 0
            sum2 = 0
            for req, effort in self._p.effort_req.items():
                e1, e2, e3, e4 = effort
                sum1 += e3 * self.x_val_nec(x, req, alpha)
                sum2 += (e4 - e3) * self.x_val_nec(x, req, alpha)
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

    def precedence_rule_pos(self, x):
        constraint_values = []

        for alpha in self._p.AC:
            for i, j in self._p.prereq:
                left_side = self.x_val_pos(x, j, alpha)
                right_side = self.x_val_pos(x, i, alpha)
                constraint_values.append(left_side - right_side)
        assert (self._p.len_ac * len(self._p.prereq)) == len(constraint_values)
        return np.array(constraint_values)

    def precedence_rule_nec(self, x):
        constraint_values = []
        for alpha in self._p.AC:
            for i, j in self._p.prereq:
                left_side = self.x_val_nec(x, j, alpha)
                right_side = self.x_val_nec(x, i, alpha)
                constraint_values.append(left_side - right_side)
        assert (self._p.len_ac * len(self._p.prereq)) == len(constraint_values)
        return np.array(constraint_values)

    def interest_rule_pos(self, x):
        constraint_values = []
        for alpha in self._p.AC:
            for customer_i, req_j in self._p.interests:
                left_side = self.y_val_pos(x, customer_i, alpha)
                right_side = self.x_val_pos(x, req_j, alpha)
                constraint_values.append(left_side - right_side)
        assert (self._p.len_ac * len(self._p.interests)) == len(constraint_values)
        return np.array(constraint_values)

    def interest_rule_nec(self, x):
        constraint_values = []
        for alpha in self._p.AC:
            for customer_i, req_j in self._p.interests:
                left_side = self.y_val_nec(x, customer_i, alpha)
                right_side = self.x_val_nec(x, req_j, alpha)
                constraint_values.append(left_side - right_side)
        assert (self._p.len_ac * len(self._p.interests)) == len(constraint_values)
        return np.array(constraint_values)

    def nested_plans_pos_rule(self, x):
        constraint_values = []
        for req in self._p.effort_req.keys():
            for alpha1 in self._p.AC:
                greater_alphas = [alpha for alpha in self._p.AC if alpha > alpha1]
                if not greater_alphas:
                    constraint_values.append(0)
                else:
                    alpha2 = min(greater_alphas)
                    left_side = self.x_val_pos(x, req, alpha2)
                    right_side = self.x_val_pos(x, req, alpha1)
                    constraint_values.append(left_side - right_side)
        assert len(constraint_values) == (self._p.len_ac * len(self._p.effort_req))
        return np.array(constraint_values)

    def nested_plans_nec_rule(self, x):
        constraint_values = []
        for req in self._p.effort_req.keys():
            for alpha1 in self._p.AC:
                greater_alphas = [alpha for alpha in self._p.AC if alpha > alpha1]
                if not greater_alphas:
                    constraint_values.append(0)
                else:
                    alpha2 = min(greater_alphas)
                    left_side = self.x_val_nec(x, req, alpha1)
                    right_side = self.x_val_nec(x, req, alpha2)
                    constraint_values.append(left_side - right_side)
        assert len(constraint_values) == (self._p.len_ac * len(self._p.effort_req))
        return np.array(constraint_values)

    def nested_plans_nec_to_pos_rule(self, x):
        max_ac = max(self._p.AC)
        constraint_values = []
        for req in self._p.effort_req.keys():
            left_side = self.x_val_nec(x, req, max_ac)
            right_side = self.x_val_nec(x, req, max_ac)
            constraint_values.append(left_side - right_side)
        assert len(constraint_values) == len(self._p.effort_req)
        return np.array(constraint_values)


    # x: 1 x NVar
    def _evaluate(self, x, out, *args, **kwargs):
        x = x.astype(int)
        res = self._calculate_obj_function(x)
        out["F"] = res

        d_pos = self.disponibility_rule_pos(x)
        d_nec = self.disponibility_rule_nec(x)
        p_pos = self.precedence_rule_pos(x)
        p_nec = self.precedence_rule_nec(x)
        i_pos = self.interest_rule_pos(x)
        i_nec = self.interest_rule_nec(x)
        n_pos = self.nested_plans_pos_rule(x)
        n_rec = self.nested_plans_nec_rule(x)
        n_nec_to_pos = self.nested_plans_nec_to_pos_rule(x)

        stacked_constraints = np.concatenate([
            d_pos, d_nec, p_pos, p_nec, i_pos, i_nec, n_pos, n_rec, n_nec_to_pos
        ])
        out["G"] = stacked_constraints


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
        selection=selection,
        crossover=crossover,
        save_history=True,
        verbose=True
    )

    X = res.X
    F = res.F
    print(f"{X=}")
    print(f"{F=}")


if __name__ == '__main__':
    main()
