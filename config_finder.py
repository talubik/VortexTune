import optuna
import time
import argparse
from optuna.exceptions import TrialPruned
import subprocess
import os
import re

N_WORKERS = None
DELAY_PER_WORKER = 60.0

storage_name = "sqlite:///configuration.db"
CORES_CHOICES = [1, 2, 4, 8]
WARP_CHOICES = [2, 4, 8, 16, 32, 64]
THREADS_CHOICES = [1, 2, 4, 8, 16, 32, 64]
CLUSTERS_CHOICES = [1, 2, 4]
L2_CACHE = [0, 1]
L3_CACHE = [0, 1]

spla_path = None
vortex_path = None
mtx_path = None


def run_vortex_simulation(cores, warps, threads, clusters, l2cache, l3cache):

    WORK_DIR = os.path.abspath(os.path.dirname(__file__))
    cmd = f"./run_vortex.sh {cores} {warps} {threads} {clusters} {l2cache} {l3cache} {vortex_path} {spla_path} {mtx_path}"
    start_time = time.time()
    proc = subprocess.run(
        cmd,
        shell=True,
        cwd=WORK_DIR,
        capture_output=True,
        text=True,
        executable="/bin/bash",
        timeout=60_000.0,
    )
    end_time = time.time()

    elapsed = end_time - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    logs_dir = os.path.join(WORK_DIR, "logs_fixed_cores_clusters")
    os.makedirs(logs_dir, exist_ok=True)
    ts = int(time.time())
    fname = f"logs_{ts}_{cores}_{warps}_{threads}_{clusters}_{l2cache}_{l3cache}.txt"
    logpath = os.path.join(logs_dir, fname)
    with open(logpath, "w") as lf:
        lf.write(f"CMD: {cmd}\nRETURN CODE: {proc.returncode}\n\n---STDOUT---\n")
        lf.write(proc.stdout or "")
        lf.write("\n\n---STDERR---\n")
        lf.write(proc.stderr or "")
        lf.write(f"\n\n---EXECUTION TIME---\n{hours}h {minutes}m\n")
    if proc.returncode != 0:
        raise TrialPruned()

    out = proc.stdout or ""
    total = 0
    for line in out.splitlines():
        if "core" in line:
            continue
        for m in re.finditer(r"cycles=([0-9]+)", line):
            try:
                total += int(m.group(1))
            except ValueError:
                continue
    return int(total)


def objective(trial):

    cores = int(trial.suggest_categorical("cores", CORES_CHOICES))
    warps = int(trial.suggest_categorical("warps", WARP_CHOICES))
    threads = int(trial.suggest_categorical("threads", THREADS_CHOICES))
    clusters = int(trial.suggest_categorical("clusters", CLUSTERS_CHOICES))
    l2cache = int(trial.suggest_categorical("l2cache", L2_CACHE))
    l3cache = int(trial.suggest_categorical("l3cache", L3_CACHE))

    stagger = (trial.number % N_WORKERS) * DELAY_PER_WORKER
    time.sleep(stagger)

    return run_vortex_simulation(cores, warps, threads, clusters, l2cache, l3cache)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=10, help="count of trials")
    parser.add_argument("--jobs", type=int, default=10, help="n_jobs")
    parser.add_argument(
        "--study-name", type=str, default="vortex-optimization2", help="study name"
    )
    parser.add_argument(
        "--spla_path", type=str, required=True, help="path to SPLA directory"
    )
    parser.add_argument(
        "--vortex_path", type=str, required=True, help="path to Vortex directory"
    )
    parser.add_argument("--mtx_path", type=str, required=True, help="path to .mtx file")
    args = parser.parse_args()
    global spla_path, vortex_path, mtx_path
    spla_path = args.spla_path
    vortex_path = args.vortex_path
    mtx_path = args.mtx_path
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        storage=storage_name,
        load_if_exists=True,
    )
    global N_WORKERS
    N_WORKERS = args.jobs
    study.optimize(objective, n_trials=args.batch, n_jobs=args.jobs)

    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)


if __name__ == "__main__":
    main()
