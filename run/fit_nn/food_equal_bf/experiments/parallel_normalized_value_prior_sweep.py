"""Opt-in CPU parallelism for the existing Food Equal normalized-value S4.

Keep the original prior, 1-ms Cython simulator, rejection/exclusion rules,
BayesFlow workflow, training schedule, output folders and S6 unchanged.
Only the independent synthetic-participant generation is parallelized.

Example:
    python -u experiments/parallel_normalized_value_prior_sweep.py \
        --sim-workers 8 --stage s4 --family ddm --mode formal --gamma-max 2

Benchmark (no training, use mode=test to keep provenance separate):
    python -u experiments/parallel_normalized_value_prior_sweep.py \
        --sim-workers 8 --benchmark-sim 128 --stage s4 --family ddm \
        --mode test --gamma-max 2

The benchmark verifies exact serial/parallel equality using identical
per-participant seeds. These seeds are NOT the legacy serial RNG stream.
Distributional equivalence does not imply identical fitted neural weights.
"""
import argparse
import copy
import ctypes
import math
import multiprocessing as mp
import os
# Set before any _shared import; worker interpreters inherit this flag.
os.environ["ATT_BF_VALUE_NORMALIZATION"] = "dataset_max"
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

_WORKER_EX = None
_WORKER_MODEL_NAMES = None
_STATS_KEYS = ("trials", "nonhits", "maxT_excluded", "rejected_subjects",
               "accepted_subjects")


def _worker_init(input_path, infos, profile, attention_priors, input_hash):
    # Spawn fresh CPU-only processes rather than forking a CUDA runtime.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    import _shared.data as data
    global _WORKER_EX, _WORKER_MODEL_NAMES
    data.model_infos.clear()
    data.model_infos.update(copy.deepcopy(infos))
    data.PRIOR_PROFILE = profile
    data.ATTENTION_PRIORS.update(attention_priors)
    _WORKER_EX = data.Experiment(input_path)
    _WORKER_MODEL_NAMES = list(data.MODEL_NAMES)
    if _WORKER_EX.sha256 != input_hash:
        raise RuntimeError("Parallel worker input differs from training input")


def _one(task):
    """One independent draw; the full participant is kept on one CPU."""
    import numpy as np
    from _shared.data import pack
    model, seed, index = task
    if _WORKER_EX is None:
        raise RuntimeError("Worker not initialized")
    root = np.random.SeedSequence([int(seed), _WORKER_MODEL_NAMES.index(model),
                                  int(index)])
    rng = np.random.default_rng(root.spawn(1)[0])
    # c_bifood uses libc rand, independent of NumPy. Each process has its
    # own libc state. Seed each participant independently of scheduling.
    cseed = int(root.spawn(1)[0].generate_state(1, dtype=np.uint32)[0])
    ctypes.CDLL("msvcrt" if sys.platform == "win32" else None).srand(cseed)
    stats = dict.fromkeys(_STATS_KEYS, 0)
    # Same sampling, retry cap, acceptance rule and packing as
    # _shared.data.Experiment.simulator(model_only=True).
    for _ in range(100):
        _, rows = _WORKER_EX.simulate(model, rng, stats)
        if len(rows) > 5:
            stats["accepted_subjects"] += 1
            return pack(rows, _WORKER_EX.max_trials), stats
        stats["rejected_subjects"] += 1
    raise RuntimeError(f"{model}: 100 rejected subjects; inspect priors")


def _batch(results, shape):
    import numpy as np
    records = [r for r, _ in results]
    if not records:
        raise ValueError("Empty simulator batch")
    data = {}
    for k in records[0]:
        x = np.stack([r[k] for r in records], axis=0)
        data[k] = x.reshape(tuple(shape) + x.shape[1:])
    return data


def _check_workers(workers):
    if workers < 1:
        raise ValueError("--sim-workers must be positive")
    # Respect the Slurm CPU allocation, reserving a CPU for the GPU process.
    limit = os.environ.get("SLURM_CPUS_PER_TASK")
    if limit and workers > max(1, int(limit) - 1):
        raise ValueError("Request --cpus-per-task >= --sim-workers + 1")


def main():
    # Parse only the acceleration arguments; pass all scientific options to
    # the existing, versioned shared_attention_prior_sweep entrypoint.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sim-workers", type=int, default=8)
    p.add_argument("--benchmark-sim", type=int, default=0,
                   help="Compare this many seeded draws serially/parallel; no training")
    extra, rest = p.parse_known_args()
    _check_workers(extra.sim_workers)
    if extra.benchmark_sim < 0:
        raise ValueError("--benchmark-sim must be nonnegative")
    if extra.benchmark_sim and ("--mode" not in rest or
                                rest[rest.index("--mode") + 1] != "test"):
        raise ValueError("Use --mode test for the non-training benchmark")
    sys.argv = [sys.argv[0], *rest]

    import numpy as np
    import _shared.common as common
    import _shared.data as data
    from experiments import normalized_value_prior_sweep as sweep

    original_train = sweep.train
    original_simulator = data.Experiment.simulator

    def parallel_train(args, classifier=False):
        if not classifier:
            return original_train(args, classifier=classifier)
        folder = common.directory(args, classifier=True)
        ex = data.Experiment(args.input)
        names = common.selected_models(args)
        # Snapshot all eight active priors AFTER the sweep wrapper applies
        # its overrides. Spawned workers must not see the BF default priors.
        initargs = (str(ex.path), copy.deepcopy(data.model_infos),
                    data.PRIOR_PROFILE, dict(data.ATTENTION_PRIORS), ex.sha256)
        workers = extra.sim_workers
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_worker_init,
                                 initargs=initargs) as pool:
            if extra.benchmark_sim:
                n = extra.benchmark_sim
                tasks = [(names[i % len(names)], args.seed, i) for i in range(n)]
                # Exact equality for the same explicit seeds. Include worker
                # startup in the parallel time to avoid overstating speed.
                t0 = time.perf_counter()
                par = list(pool.map(_one, tasks, chunksize=1))
                par_sec = time.perf_counter() - t0
                _worker_init(*initargs)
                t0 = time.perf_counter()
                serial = [_one(t) for t in tasks]
                serial_sec = time.perf_counter() - t0
                for (pr, ps), (sr, ss) in zip(par, serial):
                    if ps != ss or any(not np.array_equal(pr[k], sr[k]) for k in sr):
                        raise AssertionError("Serial/parallel seeded parity failed")
                result = dict(n=n, workers=workers, exact_seeded_parity=True,
                              serial_seconds=serial_sec, parallel_seconds=par_sec,
                              speedup=serial_sec / max(par_sec, 1e-9),
                              model_names=names, input_sha256=ex.sha256,
                              prior_spec=data.prior_spec(names))
                common.dump(folder / "parallel_benchmark.json", result)
                print("[BENCHMARK]", result, flush=True)
                return
            if (folder / "best.keras").exists() and not args.overwrite:
                raise FileExistsError(f"{folder}/best.keras exists; not overwriting")
            timing = {"workers": workers, "start_method": "spawn",
                      "seed_strategy": "per-model per-participant SeedSequence + libc srand",
                      "batches": 0, "participants": 0, "simulation_seconds": 0.0}
            common.dump(folder / "parallel_simulation.json", timing)

            def parallel_simulator(self, bf, model, seed, model_only=False):
                if not model_only:
                    return original_simulator(self, bf, model, seed, model_only)
                if self.sha256 != ex.sha256:
                    raise RuntimeError("Unexpected training input in parallel simulator")
                stats = dict.fromkeys(_STATS_KEYS, 0)
                next_index = 0

                def sample(batch_shape):
                    nonlocal next_index
                    shape = (batch_shape,) if isinstance(batch_shape, int) else tuple(batch_shape)
                    n = math.prod(shape)
                    if n < 1:
                        raise ValueError("Empty simulator batch")
                    tasks = [(model, seed, i) for i in
                             range(next_index, next_index + n)]
                    next_index += n
                    t0 = time.perf_counter()
                    results = list(pool.map(_one, tasks, chunksize=1))
                    for _, counts in results:
                        for k, v in counts.items():
                            stats[k] += v
                    out = _batch(results, shape)
                    timing["batches"] += 1
                    timing["participants"] += n
                    timing["simulation_seconds"] += time.perf_counter() - t0
                    return out

                # In BayesFlow 2.1.0 LambdaSimulator supports batched callables;
                # ModelComparisonSimulator retains its original multinomial
                # model labels and the rest of the workflow is unchanged.
                return bf.simulators.LambdaSimulator(sample, is_batched=True), stats

            data.Experiment.simulator = parallel_simulator
            t0 = time.perf_counter()
            try:
                return original_train(args, classifier=classifier)
            finally:
                data.Experiment.simulator = original_simulator
                timing["total_seconds"] = time.perf_counter() - t0
                common.dump(folder / "parallel_simulation.json", timing)
                print("[PARALLEL S4 TIMING]", timing, flush=True)

    sweep.train = parallel_train
    try:
        sweep.main()
    finally:
        sweep.train = original_train


if __name__ == "__main__":
    main()
