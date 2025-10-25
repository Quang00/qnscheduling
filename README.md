# QNScheduling

<div align="center">

![Pytest and Ruff validation](https://github.com/Quang00/QNScheduling/actions/workflows/python-app.yml/badge.svg)
![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-ffdd54?logo=python&logoColor=ffdd54)

</div>

This repository simulates scheduling on-demand heralded bipartite entanglement (EPR pairs) across a multi-hop quantum network, by using Packet Generation Attempts (PGAs) from the paper [[1]](#1).

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

## Quick Start

```bash
python -m scheduling.main
```

## Example Full command

```bash
python -m scheduling.main \
    --config configurations/network/Dumbbell.gml \
    --apps 2 \
    --inst 2 2 \
    --epr 2 2 \
    --period 10.0 10.0 \
    --hyperperiod 2 \
    --ppacket 0.05 \
    --memory 50 \
    --pswap 0.95 \
    --pgen 0.001 \
    --slot-duration 0.0001 \
    --seed 42 \
    --output results
```

## References

<a id="1">[1]</a>
T. R. Beauchamp, H. Jirovská, S. Gauthier and S. Wehner, "Extended Abstract: A Modular Quantum Network Architecture for Integrating Network Scheduling with Local Program Execution," IEEE INFOCOM 2025 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), London, United Kingdom, 2025, pp. 1-7, doi: 10.1109/INFOCOMWKSHPS65812.2025.11152936.