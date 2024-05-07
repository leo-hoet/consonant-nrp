from typing import List, Tuple, Dict, Union, OrderedDict

from pydantic import BaseModel

index = int
CustomerName = Union[int, str]
CustomerUtility = float
ReqName = Union[str, int]
TrapezoidalFuzzyNumber = Tuple[float, float, float, float]


class ConsonantNRPParameters(BaseModel):
    # alpha levels
    AC: List[float]
    # Customers names. Generally the names are just numbers
    customers: OrderedDict[CustomerName, CustomerUtility]
    # Int     interest relation, customer has interest over req
    interests: List[Tuple[CustomerName, ReqName]]
    # Max effort allowed
    p: TrapezoidalFuzzyNumber
    # Associate each req with its effort
    effort_req: Dict[ReqName, TrapezoidalFuzzyNumber]
    # technical precedence relation (i, j) <-> r_i precedes r_j
    prereq: List[Tuple[ReqName, ReqName]]

    @property
    def len_req(self):
        return len(self.effort_req)

    @property
    def len_customers(self):
        return len(self.customers)

    @property
    def len_ac(self):
        return len(self.AC)


def nrp_example_data() -> ConsonantNRPParameters:
    file = 'nrp_example_consonant.json'
    return ConsonantNRPParameters.parse_file(file)


nrp_example_data()
