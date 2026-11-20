"""Experiment layer: running episodes for any pair of agents, evaluation, baselines and training."""
import contextlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from overcooked_ai_py.agents.agent import Agent, GreedyHumanModel
from overcooked_ai_py.mdp.actions import Action

from overcooked_utils import make_env
from qlearner import Learner, QAgent

RESULTS_DIR = Path(__file__).resolve().parent / "results"


class StayAgent(Agent):
    """Baseline partner that never moves."""
    def action(self, state):
        return Action.STAY, {}


def run_episode(agent0, agent1, env, learn=True, reward_mode="sparse"):
    """
    Play one episode. Works for any two agents; learner hooks are called only if present.
    reward_mode: "sparse" = 20 per soup only, "dense" = plus the library's built-in intermediate rewards.
    The returned reward/soups always count deliveries only, so modes are comparable.
    """
    with contextlib.redirect_stdout(io.StringIO()):     # silence "Computing MotionPlanner"
        env.reset()
    agents = (agent0, agent1)
    for i, a in enumerate(agents):
        a.reset()
        a.set_agent_index(i)            # reset() wipes the index, so set it after
        if hasattr(a, "learning"):
            a.learning = learn

    ep_reward, done = 0, False
    while not done:
        joint = tuple(a.action(env.state)[0] for a in agents)
        _, sparse, done, info = env.step(joint)
        r = sparse + (sum(info["shaped_r_by_agent"]) if reward_mode == "dense" else 0)
        for a in agents:
            if hasattr(a, "observe"):
                a.observe(r)            # always: it also counts subtask steps
        ep_reward += sparse

    for a in agents:
        if hasattr(a, "end_episode"):
            a.end_episode()             # closes the last subtask; updates only if learning

    stats = {"reward": ep_reward, "soups": ep_reward // 20}
    if hasattr(agent0, "trace"):
        stats["end_reasons"] = dict(agent0.end_reasons)
        stats["completed"] = dict(Counter(o for _, _, o, _, why in agent0.trace if why == "holding"))
    return stats


def evaluate(agent0, agent1, env, n=20):
    """Mean and std soups over n episodes with learning switched off."""
    soups = [run_episode(agent0, agent1, env, learn=False)["soups"] for _ in range(n)]
    return float(np.mean(soups)), float(np.std(soups))


def eps_at(ep, n_episodes, eps_min=0.05, frac=0.6):
    """Linear decay from 1 to eps_min over the first `frac` of training, then constant."""
    return max(eps_min, 1.0 - ep / (frac * n_episodes) * (1.0 - eps_min))


def train(layout, seed, n_episodes=300, eval_every=25, n_eval=5,
          alpha=0.1, gamma=0.99, timeout=25, w=0.0, reward_mode="sparse", verbose=True):
    """Train one QAgent next to a greedy partner; evaluate with eps = 0 every `eval_every` episodes."""
    np.random.seed(seed)                 # the greedy partner may use numpy's global RNG
    with contextlib.redirect_stdout(io.StringIO()):
        env = make_env(layout)
        mlam = env.mlam
    learner = Learner(seed, alpha, gamma)
    agent = QAgent(mlam, learner, eps=1.0, timeout=timeout, w=w)
    partner = GreedyHumanModel(mlam)
    curve = {"episode": [], "mean": [], "std": []}

    def do_eval(ep):
        saved = agent.eps
        agent.eps = 0.0
        m, s = evaluate(agent, partner, env, n_eval)
        agent.eps = saved
        curve["episode"].append(ep); curve["mean"].append(m); curve["std"].append(s)
        if verbose:
            print(f"{layout} seed {seed} | ep {ep:4d} | eval soups {m:5.2f} ± {s:4.2f} "
                  f"| eps {saved:.2f} | Q entries {len(learner.q_table)}")

    do_eval(0)                           # untrained policy (all ties -> random among feasible)
    for ep in range(1, n_episodes + 1):
        agent.eps = eps_at(ep - 1, n_episodes)
        run_episode(agent, partner, env, learn=True, reward_mode=reward_mode)
        if ep % eval_every == 0:
            do_eval(ep)

    config = dict(layout=layout, seed=seed, n_episodes=n_episodes, eval_every=eval_every,
                  n_eval=n_eval, alpha=alpha, gamma=gamma, timeout=timeout, w=w,
                  reward_mode=reward_mode, eps_schedule="linear 1->0.05 over 60%")
    return {"config": config, "curve": curve, "q_size": len(learner.q_table)}, agent


def save_result(result, name, subdir="q_learner"):
    """Write a train() result (or any JSON-able dict) to results/<subdir>/<name>.json."""
    out = RESULTS_DIR / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(result, indent=1))
    return path
