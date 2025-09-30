# QNScheduling

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-ffdd54?logo=python&logoColor=ffdd54)

</div>

This repository simulates real-time scheduling and execute probabilistic jobs (entanglement generation) of a given quantum network.

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
