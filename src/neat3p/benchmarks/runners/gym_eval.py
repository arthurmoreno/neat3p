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

    result = run_neat_gym("CartPole-v1", cfg, 300, 10, seed=42, net_class=RecurrentNet)
    mean_reward = result.evaluate(n_episodes=100, seed=43)
    rewards     = result.evaluate_rewards(n_episodes=100, seed=43)
    gen_stats   = result.generation_stats   # list of per-generation dicts
"""

import random
import time

import gymnasium as gym
import numpy as np
import torch

import neat3p

try:
    import psutil as _psutil
    _PROC = _psutil.Process()
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_HAS_CUDA = torch.cuda.is_available()


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
    """Winner genome + training metadata from a NEAT gym run."""

    def __init__(
        self,
        winner,
        config,
        env_id,
        net_class,
        state_dim,
        action_dim,
        use_current_activs,
        net_kwargs,
        stats_reporter,
        wall_time_seconds: float,
        rss_before_mb: float | None,
        rss_after_mb: float | None,
        peak_gpu_mb: float | None,
    ):
        self.winner = winner
        self.config = config
        self.env_id = env_id
        self.net_class = net_class
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_current_activs = use_current_activs
        self.net_kwargs = net_kwargs
        self._stats_reporter = stats_reporter
        self.wall_time_seconds = wall_time_seconds
        self.rss_before_mb = rss_before_mb
        self.rss_after_mb = rss_after_mb
        self.peak_gpu_mb = peak_gpu_mb

    @property
    def training_rss_mb(self) -> float | None:
        if self.rss_before_mb is None or self.rss_after_mb is None:
            return None
        return max(0.0, self.rss_after_mb - self.rss_before_mb)

    @property
    def winner_nodes(self) -> int:
        return len(self.winner.nodes)

    @property
    def winner_connections(self) -> int:
        return sum(1 for c in self.winner.connections.values() if c.enabled)

    @property
    def generation_stats(self) -> list[dict]:
        """Per-generation list: generation, best, mean, std, species."""
        out = []
        for i, (best_genome, species_stats) in enumerate(
            zip(
                self._stats_reporter.most_fit_genomes,
                self._stats_reporter.generation_statistics,
            )
        ):
            # species_stats is {species_id: {genome_id: fitness}}; iterate the inner dict's
            # VALUES (fitnesses), not its keys (genome ids) — see neat-python's get_fitness_stat.
            all_fitnesses = [f for fitnesses in species_stats.values() for f in fitnesses.values()]
            out.append(
                {
                    "generation": i,
                    "best": float(best_genome.fitness),
                    "mean": float(np.mean(all_fitnesses)) if all_fitnesses else 0.0,
                    "std": float(np.std(all_fitnesses)) if all_fitnesses else 0.0,
                    "species": len(species_stats),
                }
            )
        return out

    def evaluate_rewards(self, n_episodes: int = 100, seed: int = 0) -> list[float]:
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
        return rewards

    def evaluate(self, n_episodes: int = 100, seed: int = 0) -> float:
        return float(np.mean(self.evaluate_rewards(n_episodes=n_episodes, seed=seed)))


def run_neat_gym(
    env_id: str,
    config_path: str,
    max_generations: int,
    episodes_per_genome: int,
    seed: int,
    net_class,
    use_current_activs: bool = True,
    net_kwargs: dict = None,
    verbose: bool = True,
) -> GymEvalResult:
    """Run NEAT on a Gymnasium env and return a GymEvalResult.

    verbose: if False, suppresses the StdOutReporter (useful for suite runs).
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
    if verbose:
        pop.add_reporter(neat3p.StdOutReporter(True))
    stats_reporter = neat3p.StatisticsReporter()
    pop.add_reporter(stats_reporter)

    rss_before_mb = _PROC.memory_info().rss / 1024**2 if _HAS_PSUTIL else None
    if _HAS_CUDA:
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    winner = pop.run(eval_genomes, max_generations)
    wall_time = time.perf_counter() - t0

    rss_after_mb = _PROC.memory_info().rss / 1024**2 if _HAS_PSUTIL else None
    peak_gpu_mb = torch.cuda.max_memory_allocated() / 1024**2 if _HAS_CUDA else None

    env.close()

    return GymEvalResult(
        winner=winner,
        config=config,
        env_id=env_id,
        net_class=net_class,
        state_dim=state_dim,
        action_dim=action_dim,
        use_current_activs=use_current_activs,
        net_kwargs=net_kwargs,
        stats_reporter=stats_reporter,
        wall_time_seconds=wall_time,
        rss_before_mb=rss_before_mb,
        rss_after_mb=rss_after_mb,
        peak_gpu_mb=peak_gpu_mb,
    )
