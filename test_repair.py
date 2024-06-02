import unittest

from nrp_consonant_helpers import nrp_example_data
from nrp_consonant_pymoo import ConsonantFuzzyNRP
import numpy as np

from nrprepair import NrpRepair


class RepairOperatorTest(unittest.TestCase):

    def setUp(self):
        self.params = nrp_example_data()
        self.model = ConsonantFuzzyNRP(self.params)

    def test_repair_in_nec_and_pos(self):
        X = np.zeros(self.model.get_all_vars().size)
        x_nec, x_pos = self.model.accesors.get_xs(X)

        x_nec[5:] = 1
        concatenated = np.concatenate((x_nec, x_pos))
        arr = self.model.nested_plans_nec_to_pos_rule(concatenated)
        self.assertFalse((all(x <= 0 for x in arr)))

        x_repaired = NrpRepair(params=self.params).repair(concatenated)
        arr = self.model.nested_plans_nec_to_pos_rule(x_repaired)
        self.assertTrue((all(x <= 0 for x in arr)))

    def test_repair_precedence(self):
        X = np.zeros(self.model.get_all_vars().size)
        x_nec, _ = self.model.accesors.get_xs(X)
        i, j = self.params.prereq[0]
        self.assertTrue(j > i)

        # Mutate the array
        x_nec[5:] = 1
        arr = self.model.precedence_rule_nec(x_nec)
        self.assertFalse(all(x <= 0 for x in arr))

        # Repair the array with precedence
        x_repaired = NrpRepair(params=self.params).repair(x_nec)
        arr = self.model.precedence_rule_nec(x_repaired)
        self.assertTrue(all(x <= 0 for x in arr))

    def test_repair_in_alphas(self):
        X = np.zeros(self.model.get_all_vars().size)
        x_nec, _ = self.model.accesors.get_xs(X)

        alpha_zero = self.params.AC[0]
        alpha_one = self.params.AC[1]

        first_req_name = list(self.params.effort_req.items())[0][0]

        mutated_x = self.model.accesors.x_mutate(x_nec, first_req_name, alpha_zero, 1)

        arr = self.model.nested_plans_nec_rule(mutated_x)
        self.assertFalse(all(x <= 0 for x in arr))

        # Repair the array with precedence
        x_repaired = NrpRepair(params=self.params).repair(mutated_x)
        arr = self.model.nested_plans_nec_rule(x_repaired)
        self.assertTrue(all(x <= 0 for x in arr))
