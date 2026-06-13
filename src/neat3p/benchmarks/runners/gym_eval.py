"""
Generational NEAT eval loop over a Gymnasium environment.

Two net interfaces are supported:

RecurrentNet-style (has a ``create`` classmethod):
    net = RecurrentNet.create(genome, config, batch_size=1, use_current_activs=True)
    net.reset(batch_size=1)
    out = net.activate([obs_list])          # returns tensor (1, action_dim)

nn.Module-style (NEATNetWithFeatureAttention etc.):
    net = NetClass(genome, state_dim, action_dim, config, **net_kwargs)
    net.reset(batch_size=1)
    out = net(obs_tensor)                   # forward(), returns tensor (1, action_dim)

The harness auto-detects the interface via ``hasattr(net_class, 'create')``.

Usage::

    from neat3p.benchmarks.runners.gym_eval import run_neat_gym
    from neat3p.nn.phenotypes.recurrent_net import RecurrentNet
    from neat3p.nn.composite import NEATNetWithFeatureAttention

    # RecurrentNet
    result = run_neat_gym("CartPole-v1", cfg, 300, 10, seed=42, net_class=RecurrentNet)

    # NEATNetWithFeatureAttention (feature_dim passed via net_kwargs)
    result = run_neat_gym(
        "CartPole-v1", cfg, 300, 10, seed=42,
        net_class=NEATNetWithFeatureAttention,
        net_kwargs={"feature_dim": 4},
    )

    mean_reward = result.evaluate(n_episodes=100, seed=43)
"""

import random

import gymnasium as gym
import numpy as np
import torch

import neat3p


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _is_recurrent_style(net_class) -> bool:
    return hasattr(net_class, "create")


def _make_net(net_class, genome, config, state_dim, action_dim, use_current_activs, net_kwargs):
    if _is_recurrent_style(net_class):
        return net_class.create(genome, config, batch_size=1, use_current_activs=use_current_activs)
    return net_class(genome, state_dim, action_dim, config, **net_kwargs)


def _rollout_recurrent(env, net) -> float:
    obs, _ = env.reset()
    net.reset(batch_size=1)
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        out = net.activate([obs.tolist()])
        action = int(out[0].argmax().item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
    return total


def _rollout_module(env, net) -> float:
    obs, _ = env.reset()
    net.reset(batch_size=1)
    total = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        out = net(obs_t)
        action = int(out.argmax(dim=1).item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
    return total


def _rollout(env, net, recurrent_style: bool) -> float:
    if recurrent_style:
        return _rollout_recurrent(env, net)
    return _rollout_module(env, net)


class GymEvalResult:
    """Holds the winner genome + config so the caller can run final evaluation."""

    def __init__(self, winner, config, env_id, net_class, state_dim, action_dim, use_current_activs, net_kwargs):
        self.winner = winner
        self.config = config
        self.env_id = env_id
        self.net_class = net_class
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_current_activs = use_current_activs
        self.net_kwargs = net_kwargs

    def evaluate(self, n_episodes: int = 100, seed: int = 0) -> float:
        _seed_all(seed)
        env = gym.make(self.env_id)
        recurrent_style = _is_recurrent_style(self.net_class)
        net = _make_net(
            self.net_class,
            self.winner,
            self.config,
            self.state_dim,
            self.action_dim,
            self.use_current_activs,
            self.net_kwargs,
        )
        rewards = [_rollout(env, net, recurrent_style) for _ in range(n_episodes)]
        env.close()
        return float(np.mean(rewards))


def run_neat_gym(
    env_id: str,
    config_path: str,
    max_generations: int,
    episodes_per_genome: int,
    seed: int,
    net_class,
    use_current_activs: bool = True,
    net_kwargs: dict = None,
) -> GymEvalResult:
    """Run NEAT on a Gymnasium env and return the winner wrapped in GymEvalResult.

    Seeded: seeds Python/numpy/torch so runs are reproducible.
    net_kwargs: extra kwargs forwarded to nn.Module-style nets (e.g. feature_dim=4).
                Ignored for RecurrentNet-style nets.
    """
    if net_kwargs is None:
        net_kwargs = {}

    _seed_all(seed)

    config = neat3p.Config(
        neat3p.DefaultGenome,
        neat3p.DefaultReproduction,
        neat3p.DefaultSpeciesSet,
        neat3p.DefaultStagnation,
        config_path,
    )

    env = gym.make(env_id)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    recurrent_style = _is_recurrent_style(net_class)

    def eval_genomes(genomes, cfg):
        for _gid, genome in genomes:
            net = _make_net(net_class, genome, cfg, state_dim, action_dim, use_current_activs, net_kwargs)
            genome.fitness = float(np.mean([_rollout(env, net, recurrent_style) for _ in range(episodes_per_genome)]))

    pop = neat3p.Population(config)
    pop.add_reporter(neat3p.StdOutReporter(True))
    pop.add_reporter(neat3p.StatisticsReporter())
    winner = pop.run(eval_genomes, max_generations)
    env.close()

    return GymEvalResult(winner, config, env_id, net_class, state_dim, action_dim, use_current_activs, net_kwargs)
