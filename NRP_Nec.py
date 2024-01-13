# Import
import pyomo.environ as pyo
#import pyomo as pyo

model = pyo.AbstractModel()

#model.number_of_requirements = Param(within=pyo.NonNegativeIntegers)

## Definición de conjuntos ##
#  Conjuntos
#       R       requirements
#       Prec    technical precedence relation (i, j) <-> r_i precedes r_j
#       C       customers
#       Int     interest relation


model.R = pyo.Set()
model.Prec = pyo.Set(within=model.R * model.R)
model.C = pyo.Set()
model.Int = pyo.Set(within=model.C * model.R)

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

# Nivel alpha
model.alpha = pyo.Param(within=pyo.NonNegativeReals, default=0.0, mutable=True)
#model.alpha = Var(within=NonNegativeReals, bounds=(0, 1))

# Variables de Decisión
model.x = pyo.Var(model.R, domain=pyo.Binary)
model.y = pyo.Var(model.C, domain=pyo.Binary)

def objective_rule(model):
    return sum(model.u[i] * model.y[i] for i in model.C)
    #return model.alpha


model.objective = pyo.Objective(rule=objective_rule, sense=pyo.maximize, doc='')
#model.objective = Objective(rule=regla_objetivo, sense=maximize, doc='Maximizar el nivel de compromiso respecto de los apartamientos de los puntos medios de los intervalos de las características clave')


def disponibility_rule(model):
    return model.p2 - sum(model.e3[j] * model.x[j] for j in model.R) >= (1 - model.alpha) * (model.p2 - model.p1 + sum((model.e4[j] - model.e3[j]) * model.x[j] for j in model.R))

model.disponibilidad = pyo.Constraint(rule=disponibility_rule, doc='')


# Definition of precedence constraint
def precedence_rule(model, i, j):
    return model.x[j] <= model.x[i]

model.precedence = pyo.Constraint(model.Prec, rule=precedence_rule, doc='')

def interest_rule(model, i, j):
    return model.y[i] <= model.x[j]

model.interest = pyo.Constraint(model.Int, rule=interest_rule, doc='')

def pyomo_postprocess(options=None, instance=None, results=None):
    print("Requirements selected are:")
    s = ""
    #    instance.display()
    for j in instance.x:
        if pyo.value(instance.x[j]) == 1:
            s = s + "\t{0}".format(j)
    s += "\nCustomers satisfied are:"
    for i in instance.y:
        if pyo.value(instance.y[i]) == 1:
            s = s + "\t{0}".format(i)
    print(s)
    print("El valor del funcional logrado es {0}".format(pyo.value(instance.objective)))