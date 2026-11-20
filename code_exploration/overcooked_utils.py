"""Environment layer: building the env, state -> facts (predicates), subtask -> motion goals,
and which subtasks are physically feasible."""
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Recipe
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS


ACTIONS = ["FetchOnion", "FetchDish", "PutOnionInPot", "StartCooking", "PickUpSoup", "Deliver", "Wait"]

def make_env(layout, horizon=400):
    """Overcooked env whose planner may use counters as motion goals."""
    mdp = OvercookedGridworld.from_layout_name(layout)
    params = NO_COUNTERS_PARAMS.copy()
    params["counter_goals"] = mdp.get_counter_locations()
    return OvercookedEnv.from_mdp(mdp, horizon=horizon, mlam_params=params)

def predicates(state, mdp, mp):
    """
    Convert raw game state into the abstract facts used by the arguments
    """
    n_players = len(state.players)
    facts = {"players": {}, "pots": {}, "demand": {}, "total_demand": 0}

    # Station locations: fixed for the layout, so look them up once (not per player)
    onion_loc = mdp.get_onion_dispenser_locations()
    dish_loc = mdp.get_dish_dispenser_locations()
    pot_loc = mdp.get_pot_locations()
    serve_loc = mdp.get_serving_locations()
    # Empty counters change with the state (things get placed / picked up)
    empty_counter_loc = mdp.get_empty_counter_locations(state)

    for i in range(n_players):
        player = state.players[i]
        facts["players"][i] = {}

        # Position and orientation (kept for debugging; templates should use dist)
        facts["players"][i]["position"] = player.position
        facts["players"][i]["orientation"] = player.orientation

        # Object Holding
        if player.has_object():
            holding = player.get_object().name
        else:
            holding = "nothing"
        facts["players"][i]["holding"] = holding

        # Distances (planner cost includes the final interact; np.inf = unreachable)
        start = player.pos_and_or
        facts["players"][i]["dist"] = {
            "onion": mp.min_cost_to_feature(start, onion_loc),
            "dish": mp.min_cost_to_feature(start, dish_loc),
            "pot": mp.min_cost_to_feature(start, pot_loc),
            "serve": mp.min_cost_to_feature(start, serve_loc),
            "empty_counter": mp.min_cost_to_feature(start, empty_counter_loc),
        }

    # Pots
    LABELS = {
          "empty": "empty",
          "1_items": "partial",
          "2_items": "partial",
          "3_items": "full_idle",
          "cooking": "cooking",
          "ready": "ready",
      }

    pot_states = mdp.get_pot_states(state)          # once, and from the argument
    for key, positions in pot_states.items():
        label = LABELS[key]                         # loud KeyError on unknown keys
        for pos in positions:
            if key == "empty":
                onions = 0
            else:
                onions = len(state.get_object(pos).ingredients)

            if label in ("empty", "partial"):
                demand = Recipe.MAX_NUM_INGREDIENTS - onions
            else:
                demand = 0

            facts["pots"][pos] = {"state": label, "onions": onions}
            facts["demand"][pos] = demand

    # onions already on their way reduce what's still needed
    carried = sum(1 for p in facts["players"].values() if p["holding"] == "onion")
    facts["total_demand"] = max(0, sum(facts["demand"].values()) - carried)

    return facts


def pots_for(action_name, mdp, state):
    """Pots where this subtask can be done right now (used when no specific target is given)."""
    pot_states = mdp.get_pot_states(state)
    if action_name == "PutOnionInPot":
        return mdp.get_empty_pots(pot_states) + mdp.get_partially_full_pots(pot_states)
    if action_name == "StartCooking":
        return mdp.get_full_but_not_cooking_pots(pot_states)   # not partial: avoids cooking a short soup
    if action_name == "PickUpSoup":
        return mdp.get_ready_pots(pot_states)
    raise ValueError(f"{action_name!r} is not a pot action")


def motion_goals(action_name, target, state, agent_index, mlam):
    mdp = mlam.mdp
    mplanner = mlam.motion_planner
    player = state.players[agent_index]

    counter_objects = mdp.get_counter_objects_dict(state, list(mdp.terrain_pos_dict["X"]))
    goals = []

    if action_name == "FetchOnion":
        goals = mlam.pickup_onion_actions(counter_objects)
    
    elif action_name == "FetchDish":
        goals = mlam.pickup_dish_actions(counter_objects)

    elif action_name in ("PutOnionInPot", "StartCooking", "PickUpSoup"):
        pots = [target] if target is not None else pots_for(action_name, mdp, state)
        goals = mlam._get_ml_actions_for_positions(pots)

    elif action_name == "Deliver":
        goals = mlam.deliver_soup_actions()

    else:
        raise ValueError(f"no motion mapping for {action_name}")

    # 3. keep reachable goals only
    goals = [g for g in goals if mplanner.is_valid_motion_start_goal_pair(player.pos_and_or, g)]
    
    return goals

def feasible_actions(facts, i):
    """Subtasks player i can physically do now (holding + pot state only; no strategy). Ordered as ACTIONS."""
    holding = facts["players"][i]["holding"]
    pot_states = [info["state"] for info in facts["pots"].values()]
    feasible = {"Wait"}

    if holding == "nothing":
        feasible |= {"FetchOnion", "FetchDish"}
        if any(st == "full_idle" for st in pot_states):
            feasible.add("StartCooking")   # partial pots excluded on purpose: no short soups

    elif holding == "onion":
        if any(st in ("empty", "partial") for st in pot_states):
            feasible.add("PutOnionInPot")

    elif holding == "dish":
        if any(st == "ready" for st in pot_states):
            feasible.add("PickUpSoup")

    elif holding == "soup":
        feasible.add("Deliver")

    return [a for a in ACTIONS if a in feasible]