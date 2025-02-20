from declare4pylon.choice.choice import ChoiceConstraint
from declare4pylon.choice.exclusive_choice import ExclusiveChoiceConstraint
from declare4pylon.choice.settings import ChoiceConstraintSettings
from declare4pylon.constraint import DeclareConstraint, DeclareConstraintSettings
from declare4pylon.existence.absence import AbsenceConstraint
from declare4pylon.existence.existence import ExistenceConstraint
from declare4pylon.existence.init import InitConstraint
from declare4pylon.existence.last import LastConstraint
from declare4pylon.existence.settings import ExistenceConstraintSettings
from declare4pylon.relation.existence import (
    CoExistenceConstraint,
    RespondedExistenceConstraint,
)
from declare4pylon.relation.negative import (
    NotChainSuccessionConstraint,
    NotCoExistenceConstraint,
    NotSuccessionConstraint,
)
from declare4pylon.relation.precedence import (
    AlternatePrecedenceConstraint,
    ChainPrecedenceConstraint,
    PrecedenceConstraint,
)
from declare4pylon.relation.response import (
    AlternateResponseConstraint,
    ChainResponseConstraint,
    ResponseConstraint,
)
from declare4pylon.relation.settings import RelationConstraintSettings
from declare4pylon.relation.succession import (
    AlternateSuccessionConstraint,
    ChainSuccessionConstraint,
    SuccessionConstraint,
)
from pylon.sampling_solver import WeightedSamplingSolver

from pylon_experiments.data.vocab import Vocab

constraint_string_mappings: dict[str, DeclareConstraint] = {
    "INIT": InitConstraint,
    "EXISTENCE": ExistenceConstraint,
    "ABSENCE": AbsenceConstraint,
    "LAST": LastConstraint,
    "RESPONSE": ResponseConstraint,
    "CHAIN RESPONSE": ChainResponseConstraint,
    "ALTERNATE RESPONSE": AlternateResponseConstraint,
    "PRECEDENCE": PrecedenceConstraint,
    "CHAIN PRECEDENCE": ChainPrecedenceConstraint,
    "ALTERNATE PRECEDENCE": AlternatePrecedenceConstraint,
    "SUCCESSION": SuccessionConstraint,
    "CHAIN SUCCESSION": ChainSuccessionConstraint,
    "ALTERNATE SUCCESSION": AlternateSuccessionConstraint,
    "RESPONDED EXISTENCE": RespondedExistenceConstraint,
    "COEXISTENCE": CoExistenceConstraint,
    "NOT COEXISTENCE": NotCoExistenceConstraint,
    "NOT SUCCESSION": NotSuccessionConstraint,
    "NOT CHAIN SUCCESSION": NotChainSuccessionConstraint,
    "CHOICE": ChoiceConstraint,
    "EXCLUSIVE CHOICE": ExclusiveChoiceConstraint,
}


def get_settings(
    constraint_str: str, activities: list[int]
) -> DeclareConstraintSettings:
    if constraint_str in ["EXISTENCE", "ABSENCE", "LAST", "INIT"]:
        return ExistenceConstraintSettings(activity=activities[0])
    elif constraint_str in ["CHOICE", "EXCLUSIVE CHOICE"]:
        return ChoiceConstraintSettings(a=activities[0], b=activities[1])
    else:
        return RelationConstraintSettings(a=activities[0], b=activities[1])


def constraint_from_string(string: str, vocab: Vocab) -> DeclareConstraint:
    constraint_str = string.split("[")[0].upper()
    if constraint_str not in constraint_string_mappings:
        raise ValueError(f"Constraint {constraint_str} not found.")
    constraint: type[DeclareConstraint] = constraint_string_mappings[constraint_str]

    params = string.split("[")[1].replace("]", "").split(",")

    if len(params) != 1 and len(params) != 2:
        raise ValueError(
            f"Invalid number of parameters for constraint {constraint_str}"
        )

    activities = [vocab.activity2idx[param.strip()] for param in params]
    settings = get_settings(constraint_str, activities)
    return constraint(settings=settings, solver=WeightedSamplingSolver(num_samples=100))
