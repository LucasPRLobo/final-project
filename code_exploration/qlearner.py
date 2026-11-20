"""Learning layer: abstract state, tabular SMDP Q-learner, subtask tracker and the QAgent
that executes chosen subtasks with the GreedyHumanModel planner."""
from collections import Counter, defaultdict

import numpy as np
from overcooked_ai_py.agents.agent import GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action

from overcooked_utils import feasible_actions, motion_goals, predicates
from reward_shaping import phi_values


def abstract_state(facts, i):
    """
    Return tuple of (my_holding, partners_holdings, ((pot_state, onions), ...), total_demand)
    """

    i_holding = facts['players'][i]["holding"]
    partners_holding = []
    for j in range(len(facts["players"])):
        if j != i:
            partners_holding.append(facts['players'][j]["holding"])

    pot_labels = []

    for pos, info in facts["pots"].items():
        pot_labels.append((info["state"], info['onions']))

    total_demand = facts["total_demand"]

    return (i_holding, tuple(partners_holding), tuple(sorted(pot_labels)), total_demand)



class Learner:
    def __init__(self, seed, alpha, gamma):
        self.rng = np.random.default_rng(seed)
        self.q_table = defaultdict(float)
        self.alpha = alpha
        self.gamma = gamma

    def choose(self, s, feasible, eps, phi=None):
        if phi is None:
            phi = {a: 0.0 for a in feasible}

        prob = self.rng.random()
        
        if prob < eps:
            choice = str(self.rng.choice(feasible))

        else:
            max_val = -np.inf
            best =[]

            for action in feasible:
                q = self.q_table.get((s, action), 0.0)
                shaped_q = q + phi[action]

                if shaped_q > max_val:
                    max_val = shaped_q
                    best = [action]
                elif shaped_q == max_val:
                    best.append(action)

            choice =str(self.rng.choice(best))

        return choice
    
    def update(self, s, action, G, k, s_prime, feasible_prime, phi_so=0.0, phi_next=None):
        phi_next = phi_next or {}

        target = G + (self.gamma**k) * max(
            (self.q_table.get((s_prime, o), 0.0) + phi_next.get(o, 0.0) for o in feasible_prime), 
            default=0.0)
            
        # Update
        self.q_table[(s, action)] += self.alpha * (target - phi_so - self.q_table[(s, action)])


class SubtaskTracker:
    def __init__(self, gamma):
        self.gamma = gamma
        self.subtask = None
        self.state = None

    def start(self, state, subtask, phi_o=0.0):
        self.state = state
        self.subtask = subtask
        self.G = 0
        self.k = 0
        self.phi_o = phi_o

    def add_reward(self, r):
        self.G +=  self.gamma**self.k * r
        self.k += 1

    def is_active(self):
        return self.subtask is not None
           
    def finish(self):
        output = (self.state, self.subtask, self.G, self.k, self.phi_o)
        self.subtask = None
        self.state = None

        return output


class QAgent(GreedyHumanModel):

    def __init__(self, mlam, learner, eps, w=0.0, timeout=25, **kwargs):
        # 1: store learner, eps, timeout
        self.learner = learner
        self.eps = eps
        self.w = w
        self.timeout = timeout

        # 2: self.tracker = SubtaskTracker(learner.gamma)
        self.tracker = SubtaskTracker(learner.gamma)
        self.learning = True

        # 3: self.last_holding = None, self._goals = None
        self.last_holding = None
        self._goals = None
        self.end_reasons = Counter()
        self.trace = []

        # 4: super().__init__(mlam, **kwargs) (last, as before)
        super().__init__(mlam, **kwargs)

    def reset(self):
        super().reset()
        self.tracker = SubtaskTracker(self.learner.gamma)
        self.last_holding = None
        self._goals = None
        self.end_reasons = Counter()
        self.trace = []

    def subtask_finished(self, feasible, holding):
  
        if holding != self.last_holding:
            return "holding"
        if self.tracker.subtask not in feasible:
            return "infeasible"
        if self.tracker.subtask == "Wait" and self.tracker.k >= 1:
            return "wait"
        if self.tracker.k >= self.timeout:
            return "timeout"
        
        return None

    def stay(self, state):
        self.prev_state = state      # keep greedy's unstuck bookkeeping consistent
        return Action.STAY, {"action_probs": self.a_probs_from_action(Action.STAY)}

    def action(self, state):
        # 1. READ:   facts, s, feasible, holding
        facts = predicates(state, self.mlam.mdp, self.mlam.motion_planner)
        feasible = feasible_actions(facts, self.agent_index)
        holding = facts["players"][self.agent_index]["holding"]
        s = abstract_state(facts, self.agent_index)

        phi = phi_values(facts, self.agent_index, feasible, self.w)


        # 2. END?: Close the running substack and learn from it
        reason = self.subtask_finished(feasible, holding) if self.tracker.is_active() else None

        if self.tracker.is_active() and reason is not None:
            self.end_reasons[reason] += 1
            s0, o, G, k, phi_so = self.tracker.finish()
            if self.learning:
                self.learner.update(s0, o, G, k, s, feasible, phi_so=phi_so, phi_next=phi)

            self.trace.append((state.timestep, s0, o, k, reason))

        # 3. CHOOSE: (First step, or when subtask ended)
        if not self.tracker.is_active():
            o = self.learner.choose(s, feasible, self.eps, phi)
            self.tracker.start(s, o, phi[o])
            self.last_holding = holding

        # 4. ACT
        subtask = self.tracker.subtask
        if subtask == "Wait":
            return self.stay(state)
        
        self._goals = motion_goals(subtask, None, state, self.agent_index, self.mlam)
        if not self._goals:
            return self.stay(state)
        
        return super().action(state)


    def ml_action(self, state):          # note: takes state
        return self._goals

    def observe(self, r):
        if self.tracker.is_active():
            self.tracker.add_reward(r)

    def end_episode(self):
        # TODO: if tracker active -> finish, update with feasible_prime=[]
        if self.tracker.is_active():
            s0, o, G, k, phi_so = self.tracker.finish()
            if self.learning:
                self.learner.update(s0, o, G, k, s_prime=None, feasible_prime=[], phi_so=phi_so)