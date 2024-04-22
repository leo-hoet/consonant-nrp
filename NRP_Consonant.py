import time

import matplotlib.pyplot as plt
import pyomo.environ as pyo
from tabulate import tabulate

from pyomo.opt import SolverFactory

model = pyo.AbstractModel()

## Definición de conjuntos ##
#  Conjuntos
#       R       requirements
#       Prec    technical precedence relation (i, j) <-> r_i precedes r_j
#       C       customers
#       Int     interest relation
#       AL      alpha levels


model.R = pyo.Set()
model.Prec = pyo.Set(within=model.R * model.R)
model.C = pyo.Set()
model.Int = pyo.Set(within=model.C * model.R)
model.AC = pyo.Set(within=pyo.NonNegativeReals)

# Extremo izquierdo del soporte del número difuso trapecial "disponibilidad de esfuerzo"
model.p1 = pyo.Param(within=pyo.NonNegativeReals)

# Extremo derecho del soporte del número difuso trapecial "disponibilidad de esfuerzo"
model.p4 = pyo.Param(within=pyo.NonNegativeReals)

# Extremo izquierdo del núcleo del número difuso trapecial "disponibilidad de esfuerzo"
model.p2 = pyo.Param(within=pyo.NonNegativeReals)

# Extremo derecho del núcleo del número difuso trapecial "disponibilidad de esfuerzo"
model.p3 = pyo.Param(within=pyo.NonNegativeReals)

# Utility by each customer
model.u = pyo.Param(model.C, within=pyo.NonNegativeReals)

# Extremo izquierdo del soporte del número difuso trapecial "esfuerzo del requerimiento"
model.e1 = pyo.Param(model.R, within=pyo.NonNegativeReals)

# Extremo derecho del soporte del número difuso trapecial "esfuerzo del requerimiento"
model.e4 = pyo.Param(model.R, within=pyo.NonNegativeReals)

# Extremo izquierdo del núcleo del número difuso trapecial "esfuerzo del requerimiento"
model.e2 = pyo.Param(model.R, within=pyo.NonNegativeReals)

# Extremo derecho del núcleo del número difuso trapecial "esfuerzo del requerimiento"
model.e3 = pyo.Param(model.R, within=pyo.NonNegativeReals)

# Variables de Decisión
model.x_Pos = pyo.Var(model.R, model.AC, domain=pyo.Binary, doc='')
model.x_Nec = pyo.Var(model.R, model.AC, domain=pyo.Binary, doc='')
model.y_Pos = pyo.Var(model.C, model.AC, domain=pyo.Binary, doc='')
model.y_Nec = pyo.Var(model.C, model.AC, domain=pyo.Binary, doc='')


def get_a_and_p(model):
    alphas = list(model.AC)
    alphas.sort()
    extended_alpha = [0] + alphas + alphas[-1::-1] + [0]
    p = [0.5 * (max(extended_alpha[1:i + 1]) - max(extended_alpha[0:i]) + max(extended_alpha[i:-1]) - max(
        extended_alpha[i + 1:])) for i in range(1, len(extended_alpha) - 1)]
    a = ([sum(model.u[i] * model.y_Nec[i, alpha] for i in model.C) for j, alpha in enumerate(alphas)] +
         [sum(model.u[i] * model.y_Pos[i, alpha] for i in model.C) for j, alpha in enumerate(alphas[-1::-1])])
    return p, a


def objective_rule(model):
    # Carlsson & Fuller version
    # return sum(model.u[i] * (model.y_Pos[i, alpha] + model.y_Nec[i, alpha])/2 * alpha for i in model.C for alpha in model.AC)/sum(model.AC)
    # Liu & Liu version
    p, a = get_a_and_p(model)
    return sum(map(lambda x, y: x * y, p, a))  # This is a sumproduct


model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize,
                                doc='Objective function without weighting function')


def disponibility_rule_pos(model, alpha):
    return model.p3 - sum(model.e2[j] * model.x_Pos[j, alpha] for j in model.R) >= (alpha - 1) * (
            model.p4 - model.p3 + sum((model.e2[j] - model.e1[j]) * model.x_Pos[j, alpha] for j in model.R))


model.disponibility_for_pos_relation = pyo.Constraint(model.AC, rule=disponibility_rule_pos, doc='')


def disponibility_rule_nec(model, alpha):
    return model.p2 - sum(model.e3[j] * model.x_Nec[j, alpha] for j in model.R) >= (1 - alpha) * (
            model.p2 - model.p1 + sum((model.e4[j] - model.e3[j]) * model.x_Nec[j, alpha] for j in model.R))


model.disponibility_for_nec_relation = pyo.Constraint(model.AC, rule=disponibility_rule_nec, doc='')


# Definition of precedence constraint
def precedence_rule_pos(model, i, j, alpha):
    return model.x_Pos[j, alpha] <= model.x_Pos[i, alpha]


model.precedence_pos = pyo.Constraint(model.Prec, model.AC, rule=precedence_rule_pos, doc='')


def precedence_rule_nec(model, i, j, alpha):
    return model.x_Nec[j, alpha] <= model.x_Nec[i, alpha]


model.precedence_nec = pyo.Constraint(model.Prec, model.AC, rule=precedence_rule_nec, doc='')


def interest_rule_pos(model, i, j, alpha):
    return model.y_Pos[i, alpha] <= model.x_Pos[j, alpha]


model.interest_pos = pyo.Constraint(model.Int, model.AC, rule=interest_rule_pos, doc='')


def interest_rule_nec(model, i, j, alpha):
    return model.y_Nec[i, alpha] <= model.x_Nec[j, alpha]


model.interest_nec = pyo.Constraint(model.Int, model.AC, rule=interest_rule_nec, doc='')


def nested_plans_pos_rule(model, j, alpha1):
    greater_alphas = [alpha for alpha in model.AC if alpha > alpha1]
    if not greater_alphas:
        return pyo.Constraint.Skip
    else:
        alpha2 = min(greater_alphas)
        return model.x_Pos[j, alpha2] <= model.x_Pos[j, alpha1]


model.nested_plans_pos = pyo.Constraint(model.R, model.AC, rule=nested_plans_pos_rule, doc='')


def nested_plans_nec_rule(model, j, alpha1):
    greater_alphas = [alpha for alpha in model.AC if alpha > alpha1]
    if not greater_alphas:
        return pyo.Constraint.Skip
    else:
        alpha2 = min(greater_alphas)
        return model.x_Nec[j, alpha1] <= model.x_Nec[j, alpha2]


model.nested_plans_nec = pyo.Constraint(model.R, model.AC, rule=nested_plans_nec_rule, doc='')


def nested_plans_nec_to_pos_rule(model, j):
    return model.x_Nec[j, max(model.AC)] <= model.x_Pos[j, max(model.AC)]


model.nested_plans_nec_to_pos = pyo.Constraint(model.R, rule=nested_plans_nec_to_pos_rule, doc='')


def pyomo_postprocess(options=None, instance=None, results=None):
    # print("Requirements selected per alpha level:")
    # instance.display()
    alphas = list(instance.AC)
    alphas.sort()
    alphas2 = zip(alphas[0:-1], alphas[1:])

    NR_Reqs_Pos = {alpha: {r for r in instance.R if pyo.value(instance.x_Pos[r, alpha] > 0.)} for alpha in instance.AC}
    NR_Reqs_Nec = {alpha: {r for r in instance.R if pyo.value(instance.x_Nec[r, alpha] > 0.)} for alpha in instance.AC}

    NR_Custs_Pos = {alpha: {c for c in instance.C if pyo.value(instance.y_Pos[c, alpha] > 0.)} for alpha in instance.AC}
    NR_Custs_Nec = {alpha: {c for c in instance.C if pyo.value(instance.y_Nec[c, alpha] > 0.)} for alpha in instance.AC}

    for alpha1, alpha2 in alphas2:
        if not NR_Reqs_Nec[alpha1].issubset(NR_Reqs_Nec[alpha2]):
            print('Error de composición para el nivel de necesidad {}'.format(1 - alpha1))
            exit(1)
        else:
            print(
                'Los requerimientos seleccionados para el nivel de necesidad {} son {}, y el valor de z correspondiente es {}'.format(
                    1 - alpha1, NR_Reqs_Nec[alpha1],
                    sum(instance.u[i] * pyo.value(instance.y_Nec[i, alpha1]) for i in instance.C)))
    print(
        'Los requerimientos seleccionados para el nivel de necesidad {} son {}, y el valor de z correspondiente es {}'.format(
            1 - alphas[-1], NR_Reqs_Nec[alphas[-1]],
            sum(instance.u[i] * pyo.value(instance.y_Nec[i, alphas[-1]]) for i in instance.C)))

    alphas.reverse()
    alphas2 = zip(alphas[0:-1], alphas[1:])
    for alpha1, alpha2 in alphas2:
        if not NR_Reqs_Pos[alpha1].issubset(NR_Reqs_Pos[alpha2]):
            print('Error de composición para el nivel de posibilidad {}'.format(alpha1))
            exit(1)
        else:
            print(
                'Los requerimientos seleccionados para el nivel de posibilidad {} son {}, y el valor de z correspondiente es {}'.format(
                    alpha1, NR_Reqs_Pos[alpha1],
                    sum(instance.u[i] * pyo.value(instance.y_Pos[i, alpha1]) for i in instance.C)))
    print(
        'Los requerimientos seleccionados para el nivel de posibilidad {} son {}, y el valor de z correspondiente es {}'.format(
            alphas[-1], NR_Reqs_Pos[alphas[-1]],
            sum(instance.u[i] * pyo.value(instance.y_Pos[i, alphas[-1]]) for i in instance.C)))
    print("El valor esperado de este plan consonante es {0}".format(pyo.value(instance.objective)))

    nec_r_in_plan = {r: (
        1 if not (ALPHAS := [alpha for alpha in instance.AC if pyo.value(instance.x_Nec[r, alpha]) == 1]) else min(
            ALPHAS)) for r in instance.R}
    pos_r_in_plan = {r: (
        0 if not (ALPHAS := [alpha for alpha in instance.AC if pyo.value(instance.x_Pos[r, alpha]) == 1]) else max(
            ALPHAS)) for r in instance.R}

    nec_c_in_plan = {c: (
        1 if not (ALPHAS := [alpha for alpha in instance.AC if pyo.value(instance.y_Nec[c, alpha]) == 1]) else min(
            ALPHAS)) for c in instance.C}
    pos_c_in_plan = {c: (
        0 if not (ALPHAS := [alpha for alpha in instance.AC if pyo.value(instance.y_Pos[c, alpha]) == 1]) else max(
            ALPHAS)) for c in instance.C}

    table_alpha_header = ["alpha", "Reqs for Nec", "Reqs for Pos", "Customers for Nec", "Customers for Pos"]
    table_alpha_data = [
        [alpha, ", ".join(str(r) for r in NR_Reqs_Nec[alpha]), ", ".join(str(r) for r in NR_Reqs_Pos[alpha]),
         ", ".join(str(c) for c in NR_Custs_Nec[alpha]), ", ".join(str(c) for c in NR_Custs_Pos[alpha])] for alpha in
        instance.AC]
    print(tabulate(table_alpha_data, headers=table_alpha_header))
    table_reqs_header = ["Req", "Max Pos Lvl", "Max Nec Lvl"]
    table_reqs_data = [[r, pos_r_in_plan[r], 1 - nec_r_in_plan[r]] for r in instance.R]
    print(tabulate(table_reqs_data, headers=table_reqs_header))

    table_custs_header = ["Cust", "Max Pos Lvl", "Max Nec Lvl"]
    table_custs_data = [[c, pos_c_in_plan[c], 1 - nec_c_in_plan[c]] for c in instance.C]
    print(tabulate(table_custs_data, headers=table_custs_header))

    fig, axs = plt.subplots(len(instance.AC), 2)
    for i, alpha in enumerate(instance.AC):
        axs[i, 0].set_title("Plan for Nec = 1 - {}".format(alpha))
        axs[i, 0].plot([instance.p1, instance.p2, instance.p3, instance.p4], [0, 1, 1, 0], label=r"$\tilde p$")

        e1_sum = sum(instance.e1[r] for r in instance.R if pyo.value(instance.x_Nec[r, alpha]) == 1)
        e2_sum = sum(instance.e2[r] for r in instance.R if pyo.value(instance.x_Nec[r, alpha]) == 1)
        e3_sum = sum(instance.e3[r] for r in instance.R if pyo.value(instance.x_Nec[r, alpha]) == 1)
        e4_list = [instance.e4[r] for r in instance.R if pyo.value(instance.x_Nec[r, alpha]) == 1]
        e4_sum = sum(e4_list)

        axs[i, 0].plot([e1_sum, e2_sum, e3_sum, e4_sum], [0, 1, 1, 0],
                       label=r"$\sum_{r_j \in R} \tilde e_j \cdot x_{j,{alpha}}^{Nec}$")
        axs[i, 0].plot([min(instance.p1, e1_sum) - 10, max(instance.p4, e4_sum) + 10], [alpha, alpha], '--')
        axs[i, 0].legend(loc='upper right', shadow=True)

        axs[i, 1].set_title("Plan for Pos = {}".format(alpha))
        axs[i, 1].plot([instance.p1, instance.p2, instance.p3, instance.p4], [0, 1, 1, 0], label=r"$\tilde p$")

        e1_sum = sum(instance.e1[r] for r in instance.R if pyo.value(instance.x_Pos[r, alpha]) == 1)
        e2_sum = sum(instance.e2[r] for r in instance.R if pyo.value(instance.x_Pos[r, alpha]) == 1)
        e3_sum = sum(instance.e3[r] for r in instance.R if pyo.value(instance.x_Pos[r, alpha]) == 1)
        e4_list = [instance.e4[r] for r in instance.R if pyo.value(instance.x_Pos[r, alpha]) == 1]
        e4_sum = sum(e4_list)

        axs[i, 1].plot([e1_sum, e2_sum, e3_sum, e4_sum], [0, 1, 1, 0],
                       label=r"$\sum_{r_j \in R} \tilde e_j \cdot x_{j,{alpha}}^{Pos}$")
        axs[i, 1].plot([min(instance.p1, e1_sum) - 10, max(instance.p4, e4_sum) + 10], [alpha, alpha], '--')
        axs[i, 1].legend(loc='upper right', shadow=True)
    plt.show()


if __name__ == '__main__':
    instance = model.create_instance("nrp_100c_140r_consonant.dat")

    opt = SolverFactory('cbc')
    opt.options["threads"] = 16
    t0 = time.time()

    results = opt.solve(instance)
    t1 = time.time()
    elapsed_time = t1 - t0
    print('Execution time:', elapsed_time, 'seconds')
    print(str(results))
    pyomo_postprocess(instance=instance)
