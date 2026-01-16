# QNScheduling

<div align="center">

![Pytest and Ruff validation](https://github.com/Quang00/QNScheduling/actions/workflows/python-app.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-ffdd54?logo=python&logoColor=ffdd54)

</div>

This repository simulates on-demand entanglement packet scheduling using Packet Generation Attempts (PGAs) from the paper [[1]](#1).

The simulator can:
- Generates application demands (src/dst, periods, number of packets, number of required EPR pairs)
- Computes per-application PGA “service times” from a probabilistic model
- Schedules PGAs with either a **static EDF table** or **dynamic online EDF-style** decisions
- Runs a stochastic simulation of entanglement generation/swapping with link contention, deferrals, retries, and drops
- Exports results and summary metrics as CSVs

This is the high-level workflow of the dynamic scheduler of entanglement packets:

<p align="center">
  <img src="docs/dynamic_online_scheduler_pga.png" alt="Dynamic Online Scheduler" />
</p>

## Installation

1. **Create a Python Virtual Environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
```

2. **Install the requirements**

```bash
pip install -r requirements.txt
```

## Scheduling modes and statuses

### Static scheduler (`--scheduler static`)
- Builds a precomputed EDF schedule over a horizon.
- Uses a conflict graph for concurrency: applications that share at least one link cannot overlap.
- Feasibility checks:
  - Rejects any task with utilization $U = \frac{\text{duration}}{\text{period}} > 1$
  - Rejects schedules that would miss deadlines (`end > deadline`)

If infeasible, the run exits early (no result CSVs are written for that run).

### Dynamic scheduler (`--scheduler dynamic`)
- Online EDF-style dispatching as jobs arrive.
- Arrivals are periodic by default; if `--arrival-rate` is provided, arrivals follow a Poisson process.
- Jobs may be deferred, retried, or dropped.

### Status values (in `pga_results.csv`)
- `completed`: generated required E2E EPR pairs within its budget time (PGA)
- `failed`: ran but didn’t generate enough pairs by the end of its PGA
- `retry`: failed attempt that is rescheduled again (dynamic mode)
- `defer`: could not start due to busy links, rescheduled to later (dynamic mode)
- `drop`: cannot start/finish by deadline constraints (dynamic mode)
- `missing`: expected PGA never executed/logged; filled during saving for completenes

## CLI options

Run `python -m scheduling.main --help` for the full list. Common flags:

- `--config`, `-c`: Path to a network topology `.gml` (default: `configurations/network/Dumbbell.gml`)
- `--apps`, `-a`: Number of applications to generate ($a$)
- `--inst MIN MAX`, `-i MIN MAX`: Range for instances per application ($I_a$)
- `--epr MIN MAX`, `-e MIN MAX`: Range for EPR pairs requested per application ($q_a$)
- `--period MIN MAX`, `-p MIN MAX`: Range for application periods (seconds) ($T_a$)
- `--hyperperiod`, `-hp`: Number of hyperperiod ($H_i$) cycles to schedule/simulate (primarily used by **static** scheduler)
- `--ppacket`, `-pp`: Target probability to compute PGA duration ($p_{packet}$)
- `--memory`, `-m`: Memory multiplexing number of independent link-generation trials per slot ($m$)
- `--pswap`, `-ps`: Swap success probability per trial ($p_{bsm}$)
- `--pgen`, `-pg`: Link-level EPR generation success probability per trial ($p_{gen}$)
- `--slot-duration`, `-sd`: Slot duration in seconds ($\tau$)
- `--scheduler`, `-sch`: Scheduling strategy: `static` or `dynamic`
- `--arrival-rate`, `-ar`: Mean arrival rate $\lambda$ for Poisson arrivals (**dynamic** mode). If omitted, arrivals are periodic.
- `--seed`, `-s`: RNG seed for reproducibility (NumPy)
- `--output`, `-o`: Output directory root (default: `results`)

## Output

Each run creates a new folder:

- `results/seed_<seed>/runN/`

Files written into the run folder:

- `pga_results.csv`: per-attempt logs (arrival/start/completion/waiting/status, plus merged app metadata)
- `params.csv`: parameters used for the run, plus runtime and run number
- `summary.csv`: one-row aggregate metrics (makespan, throughput, ratios, waiting stats, utilization stats, etc.)
- `summary_per_task.csv`: per-application breakdown (counts of statuses + deterministic PGA duration)
- `link_utilization.csv`: per-link busy time and utilization over the observed makespan
- `link_waiting.csv`: per-link waiting totals, average waiting time, average queue length

## Quick Start

```bash
python -m scheduling.main
```

## Example commands

### Dynamic scheduler (online)
```bash
python -m scheduling.main \
  --config configurations/network/Dumbbell.gml \
  --apps 2 \
  --inst 2 2 \
  --epr 2 2 \
  --period 10.0 10.0 \
  --ppacket 0.1 \
  --memory 50 \
  --pswap 0.95 \
  --pgen 0.001 \
  --slot-duration 0.0001 \
  --seed 42 \
  --scheduler dynamic \
  --arrival-rate 1.0 \
  --output results
```

### Static scheduler (offline)
```bash
python -m scheduling.main \
  --config configurations/network/Dumbbell.gml \
  --apps 2 \
  --inst 2 2 \
  --epr 2 2 \
  --period 10.0 10.0 \
  --hyperperiod 2 \
  --ppacket 0.1 \
  --memory 50 \
  --pswap 0.95 \
  --pgen 0.001 \
  --slot-duration 0.0001 \
  --seed 42 \
  --scheduler static \
  --output results
```

## Acknowledgements (Network Topologies)

The network topology configurations (`configurations/network/*.gml`) were obtained from **topohub** (https://github.com/piotrjurkiewicz/topohub) based on the Internet Topology Zoo, and are used here for research and simulation purposes.

## References

<a id="1">[1]</a>
T. R. Beauchamp, H. Jirovská, S. Gauthier and S. Wehner, "Extended Abstract: A Modular Quantum Network Architecture for Integrating Network Scheduling with Local Program Execution," IEEE INFOCOM 2025 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), London, United Kingdom, 2025, pp. 1-7, doi: 10.1109/INFOCOMWKSHPS65812.2025.11152936.