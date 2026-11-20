from argumentation import pipeline

def phi_values(facts, i, feasible, w):
    _, _, accepted = pipeline(facts)
    mine = {a.action_name for a in accepted if a.agent_idx == i}
    
    return {o: (w if o in mine else 0.0) for o in feasible}