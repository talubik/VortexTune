# Vortex Configuration Optimizer

This project provides an automated tool to explore and optimize the performance of the **Vortex GPGPU** architecture.

By utilizing **Optuna**, this script runs simulations across various hardware configurations to identify the optimal setup that minimizes execution cycles for a given workload (specifically SPLA benchmarks with MTX inputs).

## Features

- **Automated Design Space Exploration**: Automatically tunes hardware parameters to find the best performance.
- **Parallel Execution**: Supports running multiple simulation workers concurrently.
- **Persistent Storage**: Uses SQLite to save study progress, allowing you to pause and resume optimization.
- **Detailed Logging**: Captures stdout/stderr and execution time for every simulation trial.

## Tunable Parameters

The optimizer explores the following hardware configuration space:

| Parameter    | Description        | Values Explored        |
| :----------- | :----------------- | :--------------------- |
| **Cores**    | Number of cores    | 1, 2, 4, 8             |
| **Clusters** | Number of clusters | 1, 2, 4                |
| **Warps**    | Warps per core     | 2, 4, 8, 16, 32, 64    |
| **Threads**  | Threads per warp   | 1, 2, 4, 8, 16, 32, 64 |
| **L2 Cache** | L2 Cache status    | On (1), Off (0)        |
| **L3 Cache** | L3 Cache status    | On (1), Off (0)        |

## Prerequisites

1. **Python 3.x**
2. **Optuna**: Install via pip:
   ```bash
   pip install optuna
   ```
3. **Vortex & SPLA**: You must have the Vortex GPGPU simulator and SPLA benchmark compiled and available on your system.
4. **Helper Script**: Ensure `run_vortex.sh` is present in the same directory as the python script and is executable (`chmod +x run_vortex.sh`).

## Usage

Run the `config_finder.py` script with the required paths to your Vortex installation, SPLA directory, and the input matrix file.

### Basic Command

```bash
python3 config_finder.py \
  --spla_path /path/to/spla_root \
  --vortex_path /path/to/vortex_root \
  --mtx_path /path/to/input_matrix.mtx
```

### Advanced Options

You can control the number of trials and parallel jobs:

```bash
python3 config_finder.py \
  --spla_path /home/user/spla \
  --vortex_path /home/user/vortex \
  --mtx_path /home/user/data/matrix.mtx \
  --batch 100 \
  --jobs 10 \
  --study-name "vortex_optimization_experiment_1"
```

### Arguments

- `--spla_path`: (Required) Path to the SPLA directory.
- `--vortex_path`: (Required) Path to the Vortex directory.
- `--mtx_path`: (Required) Path to the `.mtx` input file.
- `--batch`: Number of trials to run (default: 10).
- `--jobs`: Number of parallel workers (default: 10).
- `--study-name`: Name of the Optuna study (default: "vortex-optimization2").

## Output

1. **Console**: Displays the best parameters and best execution cycle count found so far.
2. **Database**: Results are stored in `configuration.db` (SQLite).
3. **Logs**: Detailed logs for each simulation are saved in the `logs_fixed_cores_clusters/` directory.