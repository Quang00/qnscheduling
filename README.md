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
python -m scheduling.main run
```

## Example Full command

```bash
python -m scheduling.main run --config configurations/network/Garr201201.gml --apps 5 --inst 2 2 --epr 2 2 --period 10.0 10.0 --seed 42 --output results
```
