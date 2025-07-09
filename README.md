# QNScheduling

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
python -m scheduling.main run --config configurations/toy_example.yaml
```
