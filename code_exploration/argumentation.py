"""Argumentation layer: argument templates, attacks (R1/R2 + capacity), value filter,
tie-break and grounded semantics. Pure Python over the `facts` dict from
overcooked_utils.predicates — no Overcooked imports needed here."""
import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Argument:
    agent_idx: int
    action_name: str
    target: tuple[int, int] | None
    value_name: str
    distance: float

@dataclass(frozen=True)
class ArgTemplate:
    name: str
    premise: Callable[[dict, int], list]
    value_name: str
    dist_key: str


VALUE_RANK = {
    "V_deliver": 4,
    "V_serve":   3,
    "V_cook":    2,
    "V_prep":    1,
}


# T1  holding(i)=nothing ∧ total_demand>0                 → i: FetchOnion        | V_prep
def fetch_onion(facts, i):
    holding = facts["players"][i]['holding']
    total_demand = facts["total_demand"]
    if holding == "nothing" and total_demand > 0:
        return [None]
    else:
        return []
    
T1 = ArgTemplate(
    name="FetchOnion",
    premise=fetch_onion,
    value_name="V_prep",
    dist_key="onion"
)

# T2  holding(i)=onion   ∧ pot_state(p)∈{empty, partial}   → i: PutOnionInPot(p)  | V_prep
def put_onion_in_pot(facts, i):
    holding = facts["players"][i]['holding']
    pot_locations = []

    if holding == "onion":
        for pos, info in facts["pots"].items():
            if info["state"] in ["empty", "partial"]:
                pot_locations.append(pos)
    
    return pot_locations

T2 = ArgTemplate(
    name="PutOnionInPot",
    premise=put_onion_in_pot,
    value_name="V_prep",
    dist_key="pot"
)
    
# T3  holding(i)=nothing ∧ pot_state(p)=full_idle          → i: StartCooking(p)   | V_cook
def start_cooking(facts, i):
    holding = facts["players"][i]['holding']
    pot_locations = []

    if holding == "nothing":
        for pos, info in facts["pots"].items():
            if info["state"] == "full_idle":
                pot_locations.append(pos)

    return pot_locations

T3 = ArgTemplate(
    name="StartCooking",
    premise=start_cooking,
    value_name="V_cook",
    dist_key="pot"
)

# T4  holding(i)=nothing ∧ pot_state(p)∈{cooking, ready}   → i: FetchDish         | V_serve
def fetch_dish(facts, i):
    holding = facts["players"][i]['holding']
    if holding == "nothing":
        for info in facts["pots"].values():
            if info["state"] in ["cooking", "ready"]:
                return [None]
    
    return []
            
T4 = ArgTemplate(
    name="FetchDish",
    premise=fetch_dish,
    value_name="V_serve",
    dist_key="dish"
)

# T4' holding(i)=nothing ∧ pot_state(p)=partial ∧ total_demand=0 ∧ ∀j: holding(j)≠dish   → i: FetchDish         | V_serve
def fetch_dish_prime(facts, i):
    holding = facts["players"][i]['holding']
    total_demand = facts["total_demand"]
    someone_has_dish = any(p["holding"] == "dish" for p in facts["players"].values())

    if holding == "nothing" and total_demand == 0 and not someone_has_dish:
        for info in facts["pots"].values():
            if info["state"] == "partial":
                return [None]

    return []

T4p = ArgTemplate(
    name="FetchDish",
    premise=fetch_dish_prime,
    value_name="V_serve",
    dist_key="dish"
)


# T5  holding(i)=dish    ∧ pot_state(p)=ready              → i: PickUpSoup(p)     | V_serve
def pick_up_soup(facts, i):
    holding = facts["players"][i]['holding']
    pot_locations = []

    if holding == "dish":
        for pos, info in facts["pots"].items():
            if info["state"] == "ready":
                pot_locations.append(pos)

    return pot_locations

T5 = ArgTemplate(
    name="PickUpSoup",
    premise=pick_up_soup,
    value_name="V_serve",
    dist_key="pot"
)

# T6  holding(i)=soup   → i: Deliver   | V_deliver
def deliver(facts, i):
    holding = facts["players"][i]['holding']
    if holding == "soup":
        return [None]
    else:
        return []
    

T6 = ArgTemplate(
    name="Deliver",
    premise=deliver,
    value_name="V_deliver",
    dist_key="serve"
)


templates = [T1, T2, T3, T4, T4p, T5, T6]

def instantiate(facts, templates):
    """Apply every template to every player; one Argument per target the premise returns."""
    arguments = set()
    for i in facts["players"]:
        for template in templates:
            for target in template.premise(facts, i):
                arguments.add(Argument(agent_idx=i,
                                       action_name=template.name,
                                       target=target,
                                       value_name=template.value_name,
                                       distance=facts["players"][i]["dist"][template.dist_key]))
    return arguments



def capacity(action_name, target, facts):
    """How many players can usefully take this job right now (R2 fires when claimants exceed it)."""
    if action_name == "FetchOnion":
        # onions still needed overall, already net of onions being carried
        return facts["total_demand"]

    if action_name == "PutOnionInPot":
        # how many more onions this pot can take; the claimants ARE the carriers,
        # so use per-pot demand, not total_demand (which already subtracted them)
        return facts["demand"][target]
        
    if action_name in ("StartCooking", "PickUpSoup"):
        # one press starts a pot; one dish takes the soup
        return 1
    
    if action_name == "FetchDish":
        # pots that will need a dish soon, minus dishes already in someone's hands
        pots_needing_dish = sum(
            1 for info in facts["pots"].values()
            if info["state"] in ("cooking", "ready")
            or (info["state"] in ("partial", "full_idle") and facts["total_demand"] == 0)
        )   
        dishes_held = sum(1 for p in facts["players"].values() if p["holding"] == "dish")
        return max(0, pots_needing_dish - dishes_held)
        
    if action_name == "Deliver":
        # each player delivers their own soup: never a shared job
        return float("inf")

    raise ValueError(f"no capacity rule for action {action_name!r}")


def attacks(arguments, facts):
    attack_list = set()
    # how many arguments claim each job (action + target), counted once up front
    claimants = Counter((a.action_name, a.target) for a in arguments)

    for a, b in itertools.permutations(arguments, 2):
        # Check R1: One Player, one job at a time
        if a.agent_idx == b.agent_idx:
            attack_list.add((a,b))
            continue

        # Check R2
        same_job = (a.action_name == b.action_name) and (a.target == b.target)
        if same_job:
            job = (a.action_name, a.target)
            if claimants[job] > capacity(a.action_name, a.target, facts):
                attack_list.add((a,b))
                
        
    return attack_list


def value_filter(attack_set, value_rank=VALUE_RANK):
    """VAF step: an attack fails if the target promotes a strictly higher value than the attacker."""
    return {
        (a, b) for (a, b) in attack_set
        if value_rank[b.value_name] <= value_rank[a.value_name]
    }

def loses_tie(a, b):
    """True if argument a should give way to b in an equal-value mutual clash."""
    if a.distance != b.distance:
        return a.distance > b.distance            # farther gives way
    if a.agent_idx != b.agent_idx:
        return a.agent_idx > b.agent_idx          # convention: lower index wins
    return (a.action_name, str(a.target)) > (b.action_name, str(b.target))   # same player: fixed order
    
def tie_break(attack_set):
    """Turn equal-value mutual attacks into one-way attacks from the winner to the loser."""
    return {    
        (a, b) for (a, b) in attack_set
        if not ((b, a) in attack_set
                and a.value_name == b.value_name
                and loses_tie(a, b))
    }


# Grounded Extensions
def grounded_extension(args, attacks): # attacks are (attacker, target) pairs
    IN = set()
    OUT = set()

    attackers = {a: set() for a in args}

    for attacker, target in attacks:
        attackers[target].add(attacker)

    while True:
        last_in = set(IN)
        last_out = set(OUT)

        for arg in args:
            if set(attackers[arg]).issubset(OUT):
                IN.add(arg)

        for attack in attacks:
            if attack[0] in IN:
                OUT.add(attack[1])

        if last_in == IN and last_out == OUT:
            break

        
    return IN


def pipeline(facts, templates=templates, use_values=True, use_tie_break=True, value_rank=VALUE_RANK):
    """facts -> (arguments, attacks, grounded extension). Flags switch layers off for ablations."""
    args = instantiate(facts, templates)
    # a job nobody needs (e.g. a second dish) gets no argument, even with a single claimant
    args = {a for a in args if capacity(a.action_name, a.target, facts) >= 1}
    att = attacks(args, facts)
    if use_values:
        att = value_filter(att, value_rank)
    if use_tie_break:
        att = tie_break(att)
    return args, att, grounded_extension(args, att)