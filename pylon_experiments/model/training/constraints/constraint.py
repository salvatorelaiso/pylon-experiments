from declare4pylon.choice.choice import ChoiceConstraint
from declare4pylon.choice.exclusive_choice import ExclusiveChoiceConstraint
from declare4pylon.choice.settings import ChoiceConstraintSettings
from declare4pylon.constraint import DeclareConstraint, DeclareConstraintSettings
from declare4pylon.existence.absence import AbsenceConstraint
from declare4pylon.existence.existence import ExistenceConstraint
from declare4pylon.existence.init import InitConstraint
from declare4pylon.existence.last import LastConstraint
from declare4pylon.existence.settings import (
    ExistenceConstraintSettings,
    ExistenceCountConstraintSettings,
)
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

choice_constraints_mappings: dict[
    str, tuple[type[ChoiceConstraint], type[DeclareConstraintSettings]]
] = {
    "CHOICE": (ChoiceConstraint, ChoiceConstraintSettings),
    "EXCLUSIVE CHOICE": (ExclusiveChoiceConstraint, ChoiceConstraintSettings),
}

existence_constraints_mappings: dict[
    str, tuple[type[ExistenceConstraint], type[DeclareConstraintSettings]]
] = {
    "ABSENCE": (AbsenceConstraint, ExistenceCountConstraintSettings),
    "EXACTLY": (ExistenceConstraint, ExistenceCountConstraintSettings),
    "EXISTENCE": (ExistenceConstraint, ExistenceConstraintSettings),
    "INIT": (InitConstraint, ExistenceConstraintSettings),
    "LAST": (LastConstraint, ExistenceConstraintSettings),
}

relation_constraints_mappings: dict[
    str, tuple[type[RelationConstraintSettings], type[DeclareConstraintSettings]]
] = {
    "CO-EXISTENCE": (CoExistenceConstraint, RelationConstraintSettings),
    "RESPONDED EXISTENCE": (RespondedExistenceConstraint, RelationConstraintSettings),
    "PRECEDENCE": (PrecedenceConstraint, RelationConstraintSettings),
    "ALTERNATE PRECEDENCE": (AlternatePrecedenceConstraint, RelationConstraintSettings),
    "CHAIN PRECEDENCE": (ChainPrecedenceConstraint, RelationConstraintSettings),
    "RESPONSE": (ResponseConstraint, RelationConstraintSettings),
    "ALTERNATE RESPONSE": (AlternateResponseConstraint, RelationConstraintSettings),
    "CHAIN RESPONSE": (ChainResponseConstraint, RelationConstraintSettings),
    "SUCCESSION": (SuccessionConstraint, RelationConstraintSettings),
    "ALTERNATE SUCCESSION": (AlternateSuccessionConstraint, RelationConstraintSettings),
    "CHAIN SUCCESSION": (ChainSuccessionConstraint, RelationConstraintSettings),
    "NOT COEXISTENCE": (NotCoExistenceConstraint, RelationConstraintSettings),
    "NOT SUCCESSION": (NotSuccessionConstraint, RelationConstraintSettings),
    "NOT CHAIN SUCCESSION": (NotChainSuccessionConstraint, RelationConstraintSettings),
}

constraint_string_mappings: dict[
    tuple[type[DeclareConstraint]], type[DeclareConstraintSettings]
] = {
    **choice_constraints_mappings,
    **existence_constraints_mappings,
    **relation_constraints_mappings,
}


def constraint_from_string(
    string: str, vocab: Vocab
) -> tuple[DeclareConstraint, float]:
    string, *multiplier_str = string.split("*")
    if len(multiplier_str) == 0:
        multiplier = 1.0
    elif len(multiplier_str) == 1:
        multiplier = float(multiplier_str[0])
    else:
        raise ValueError("Invalid multiplier")

    constraint_str = string.split("[")[0].upper()
    # If `constraint_str` ends with a number, e.g. "ABSCENCE2", extract the number and remove it from the string
    count = None
    if constraint_str[-1].isdigit():
        constraint_str, count = constraint_str[:-1], int(constraint_str[-1])

    if constraint_str not in constraint_string_mappings:
        raise ValueError(f"Constraint {constraint_str} not found.")

    constraint_type: type[DeclareConstraint]
    settings_type: type[DeclareConstraintSettings]
    constraint_type, settings_type = constraint_string_mappings[constraint_str]

    params = string.split("[")[1].replace("]", "").split(",")

    if len(params) != 1 and len(params) != 2:
        raise ValueError(
            f"Invalid number of parameters for constraint {constraint_str}"
        )

    activities = [vocab.activity2idx[param.strip()] for param in params]

    if len(params) == 1:
        settings = (
            settings_type(activity=activities[0])
            if count is None
            else settings_type(activity=activities[0], count=count)
        )
    else:
        settings = (
            settings_type(a=activities[0], b=activities[1])
            if count is None
            else settings_type(a=activities[0], b=activities[1], count=count)
        )

    return (
        constraint_type(
            settings=settings, solver=WeightedSamplingSolver(num_samples=100)
        ),
        multiplier,
    )
