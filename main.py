from NRP_Pos import model as NRP_Pos_abstract_model
from NRP_Nec import model as NRP_Nec_abstract_model
from pyomo.environ import *
from pyomo.opt import SolverFactory
import matplotlib.pyplot as plt

def create_NRP_POS_instance(data_file_path):
    instance = NRP_Pos_abstract_model.create_instance(data_file_path)
    return instance

def create_NRP_NEC_instance(data_file_path):
    instance = NRP_Nec_abstract_model.create_instance(data_file_path)
    return instance


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    instance_pos = create_NRP_POS_instance("nrp_example.dat")
    instance_nec = create_NRP_NEC_instance("nrp_example.dat")

    opt = SolverFactory("cbc")

    alpha_vector = [0.2, 0.4, 0.6, 0.8, 1.0]
    z_pos_vec = []
    z_nec_vec = []

    for alpha in alpha_vector:
        instance_pos.alpha = alpha

        results_pos = opt.solve(instance_pos)
        results_nec = opt.solve(instance_nec)

        z_pos_vec.append(value(instance_pos.objective))
        z_nec_vec.append(value(instance_nec.objective))

        print("El par correspondiente al nivel {} es ({}, {})".format(alpha, value(instance_nec.objective), value(instance_pos.objective)))
        # print("Requerimientos seleccionados para Nec: {}".format({r for r in instance_nec.R if value(instance_nec.x[r]) > 0}))
        # print("Requerimientos seleccionados para Pos: {}".format({r for r in instance_pos.R if value(instance_pos.x[r]) > 0}))
        print()
    plt.scatter(z_pos_vec, alpha_vector)
    plt.scatter(z_nec_vec, alpha_vector)
    plt.show()
