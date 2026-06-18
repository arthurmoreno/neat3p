"""
Group A unit tests — behaviour-invariant through the benchmark refactor.

These test pure functions and env logic that the refactor *relocates but must not change*.
All tests run in < ~1s: no NEAT training loop, no benchmark run, no gym rollout.
"""

import os
import pickle

import pytest

# ---------------------------------------------------------------------------
# gym_eval — _seed_for, _world_seeds_pure
# ---------------------------------------------------------------------------


def test_seed_for_determinism():
    from benchmarks.runners.gym_eval import _seed_for

    assert _seed_for(42, 0) == _seed_for(42, 0)
    assert _seed_for(42, 1) != _seed_for(42, 0)


def test_seed_for_disjoint_bases():
    """Distinct bases should produce non-overlapping streams for reasonable idx ranges."""
    from benchmarks.runners.gym_eval import _seed_for

    base_a = {_seed_for(42, i) for i in range(100)}
    base_b = {_seed_for(99, i) for i in range(100)}
    # They can collide by chance but shouldn't fully overlap
    assert len(base_a & base_b) < 10


def test_world_seeds_pure_per_generation():
    from benchmarks.runners.gym_eval import _world_seeds_pure

    s0 = _world_seeds_pure("per_generation", 42, 4, 0)
    s1 = _world_seeds_pure("per_generation", 42, 4, 1)
    assert s0 is not None and len(s0) == 4
    assert s1 is not None and len(s1) == 4
    assert s0 != s1, "different generations must give different seeds"


def test_world_seeds_pure_fixed():
    from benchmarks.runners.gym_eval import _world_seeds_pure

    s_gen0 = _world_seeds_pure("fixed", 42, 4, 0)
    s_gen5 = _world_seeds_pure("fixed", 42, 4, 5)
    assert s_gen0 == s_gen5, "fixed strategy: same seeds every generation"


def test_world_seeds_pure_random():
    from benchmarks.runners.gym_eval import _world_seeds_pure

    assert _world_seeds_pure("random", 42, 4, 0) is None


# ---------------------------------------------------------------------------
# substrates — voxel_forage_substrate
# ---------------------------------------------------------------------------


def test_voxel_forage_substrate_input_count():
    from neat3p.gym_envs.substrates import voxel_forage_substrate

    in_c, hid_c, out_c = voxel_forage_substrate()
    # Default (3, 3, 3, 3) patch = 3*3*3*3 = 81 voxels + 2 scalars = 83
    assert len(in_c) == 83


def test_voxel_forage_substrate_shapes():
    from neat3p.gym_envs.substrates import voxel_forage_substrate

    in_c, hid_c, out_c = voxel_forage_substrate()
    assert all(len(c) == 2 for c in in_c), "each input coord must be [x, y]"
    assert all(len(c) == 2 for c in hid_c)
    assert all(len(c) == 2 for c in out_c)
    assert len(out_c) == 7, "VoxelForage has 7 actions"


def test_voxel_forage_substrate_hidden_count():
    from neat3p.gym_envs.substrates import voxel_forage_substrate

    _, hid_c, _ = voxel_forage_substrate(n_hidden=9)
    assert len(hid_c) == 9


def test_voxel_forage_substrate_determinism():
    from neat3p.gym_envs.substrates import voxel_forage_substrate

    a = voxel_forage_substrate()
    b = voxel_forage_substrate()
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


# ---------------------------------------------------------------------------
# artifacts — _env_tag, save_winner round-trip
# ---------------------------------------------------------------------------


def test_env_tag_scent():
    from benchmarks.artifacts import _env_tag

    assert _env_tag("VoxelForage-v0") == "scent"


def test_env_tag_noscent():
    from benchmarks.artifacts import _env_tag

    assert _env_tag("VoxelForage-NoScent-v0") == "noscent"


class _FakeGenome:
    fitness = 95.0
    nodes = {0: None, 1: None, 2: None}
    connections = {}


class _FakeResult:
    state_dim = 83
    action_dim = 7
    validation_stats = []
    wall_time_seconds = 1.0
    training_rss_mb = None
    peak_gpu_mb = None
    winner_nodes = 3
    winner_connections = 0
    generation_stats = [{"generation": 0, "best": 95.0, "mean": 50.0, "std": 10.0, "species": 1}]

    def __init__(self, genome):
        self.winner = genome


def _make_fake_genome():
    return _FakeGenome()


def _make_fake_result(genome):
    return _FakeResult(genome)


def test_save_winner_filename(tmp_path):
    from benchmarks.artifacts import save_winner

    genome = _make_fake_genome()
    result = _make_fake_result(genome)
    cfg = "/fake/config.cfg"
    path = save_winner(str(tmp_path), "recurrent_net", result, "VoxelForage-v0", 42, cfg)
    assert os.path.basename(path) == "recurrent_net_scent_seed42.pkl"
    assert os.path.isfile(path)


def test_save_winner_keys(tmp_path):
    from benchmarks.artifacts import save_winner

    genome = _make_fake_genome()
    result = _make_fake_result(genome)
    path = save_winner(str(tmp_path), "recurrent_net", result, "VoxelForage-v0", 42, "/fake.cfg")
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    for key in ("kind", "env_id", "config_path", "state_dim", "action_dim", "winner_genome", "fitness", "seed"):
        assert key in pkg, f"missing key: {key}"
    assert pkg["kind"] == "recurrent_net"
    assert pkg["fitness"] == 95.0


# ---------------------------------------------------------------------------
# report helpers — _group_by_name, _summary_table_rows, _fig_validation
# ---------------------------------------------------------------------------


def _fake_results(n_benchmarks=2, runs_each=2):
    results = []
    names = [f"bench_{i}" for i in range(n_benchmarks)]
    for name in names:
        for seed in range(runs_each):
            results.append(
                {
                    "benchmark_name": name,
                    "seed": seed,
                    "solve_generation": seed if seed > 0 else None,
                    "total_generations": 10,
                    "winner_fitness": 80.0 + seed,
                    "final_mean_reward": 75.0 + seed,
                    "final_std_reward": 5.0,
                    "wall_time_seconds": 10.0,
                    "training_rss_mb": None,
                    "peak_gpu_mb": None,
                    "winner_nodes": 5,
                    "winner_connections": 3,
                    "generation_stats": [
                        {"generation": i, "best": float(i * 10), "mean": float(i * 5)} for i in range(10)
                    ],
                    "validation_stats": [],
                    "winner_path": f"/tmp/{name}_seed{seed}.pkl",
                    "env_id": "FakeEnv-v0",
                }
            )
    return results


def test_group_by_name():
    from benchmarks.report import _group_by_name

    results = _fake_results(2, 3)
    groups = _group_by_name(results)
    assert set(groups.keys()) == {"bench_0", "bench_1"}
    assert all(len(v) == 3 for v in groups.values())


def test_summary_table_rows_keys():
    from benchmarks.report import _summary_table_rows

    results = _fake_results(2, 2)
    rows = _summary_table_rows(results)
    assert len(rows) == 2
    assert "Benchmark" in rows[0]
    assert "Median reward" in rows[0]
    assert "Solved %" in rows[0]


def test_fig_validation_none_when_absent():
    from benchmarks.report import _fig_validation

    results = _fake_results(1, 2)
    colors = {"bench_0": "#636EFA"}
    fig = _fig_validation(results, colors)
    assert fig is None, "should return None when validation_stats is empty everywhere"


def test_fig_validation_returns_figure_when_present():
    from benchmarks.report import _fig_validation

    results = _fake_results(1, 2)
    for r in results:
        r["validation_stats"] = [{"generation": i, "val_mean": float(i)} for i in range(5)]
    colors = {"bench_0": "#636EFA"}
    fig = _fig_validation(results, colors)
    assert fig is not None


# ---------------------------------------------------------------------------
# runner — stats_dict contract
# ---------------------------------------------------------------------------


def test_stats_dict_keys():
    from benchmarks.runner import stats_dict

    genome = _make_fake_genome()
    result = _make_fake_result(genome)
    rewards = [80.0, 85.0, 90.0]

    d = stats_dict(
        task_name="voxel_forage",
        model_kind="recurrent_net",
        result=result,
        rewards=rewards,
        env_id="VoxelForage-v0",
        solve_threshold=110.0,
        winner_path="/tmp/fake.pkl",
        seed=42,
    )
    required_keys = {
        "benchmark_name",
        "env_id",
        "winner_path",
        "validation_stats",
        "seed",
        "solve_generation",
        "total_generations",
        "winner_fitness",
        "final_mean_reward",
        "final_std_reward",
        "wall_time_seconds",
        "training_rss_mb",
        "peak_gpu_mb",
        "winner_nodes",
        "winner_connections",
        "generation_stats",
    }
    assert required_keys <= set(d.keys()), f"missing: {required_keys - set(d.keys())}"


def test_stats_dict_solve_gen_none_when_below_threshold():
    from benchmarks.runner import stats_dict

    genome = _make_fake_genome()
    result = _make_fake_result(genome)
    d = stats_dict(
        task_name="voxel_forage",
        model_kind="recurrent_net",
        result=result,
        rewards=[50.0],
        env_id="VoxelForage-v0",
        solve_threshold=110.0,
        winner_path="/tmp/fake.pkl",
        seed=42,
    )
    assert d["solve_generation"] is None


# ---------------------------------------------------------------------------
# tasks / models registries (Group B — new seams)
# ---------------------------------------------------------------------------


def test_tasks_registry_keys():
    from benchmarks.tasks import TASKS

    assert "voxel_forage" in TASKS
    assert "cartpole" in TASKS
    assert "lunarlander" in TASKS


def test_models_registry_keys():
    from benchmarks.models import MODELS

    for kind in ("recurrent_net", "feature_attention", "hyper_neat", "adaptive_hyperneat"):
        assert kind in MODELS, f"missing model: {kind}"


def test_voxel_forage_task_variants():
    from benchmarks.tasks import TASKS

    t = TASKS["voxel_forage"]
    assert "scent" in t.variants
    assert "noscent" in t.variants
    assert t.variants["scent"] == "VoxelForage-v0"
    assert t.variants["noscent"] == "VoxelForage-NoScent-v0"


def test_config_path_convention(tmp_path, monkeypatch):
    """config_path() falls back to <task>.cfg when <task>_<model>.cfg doesn't exist."""
    from benchmarks.models import MODELS
    from benchmarks.tasks import TASKS

    # cartpole + recurrent_net uses cartpole.cfg (no cartpole_recurrent_net.cfg)
    task = TASKS["cartpole"]
    adapter = MODELS["recurrent_net"]
    cfg = adapter.config_path(task)
    assert os.path.isfile(cfg), f"config not found: {cfg}"
    assert "cartpole" in os.path.basename(cfg)


def test_feature_attention_tunables():
    from benchmarks.models import MODELS

    fa = MODELS["feature_attention"]
    assert "pretrain_episodes" in fa.tunables
    assert "pretrain_epochs" in fa.tunables
    # accepts() filters: only declared tunables pass through
    filtered = fa.accepts({"pretrain_episodes": 50, "generations": 30, "other": True})
    assert "pretrain_episodes" in filtered
    assert "generations" not in filtered
    assert "other" not in filtered


def test_hyperneat_requires_substrate():
    from benchmarks.models import MODELS
    from benchmarks.tasks import TASKS

    hn = MODELS["hyper_neat"]
    # cartpole has no substrate — should raise
    task_no_sub = TASKS["cartpole"]
    with pytest.raises(ValueError, match="substrate"):
        hn.build(task_no_sub, "CartPole-v1", 42, "cpu", False)
