.. qnscheduling documentation master file, created by
   sphinx-quickstart on Sat Jan 31 12:34:41 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: logo.png
   :align: center
   :alt: QNScheduling Logo
   :width: 500px

.. raw:: html

   <div align="center">
      <img src="https://github.com/Quang00/QNScheduling/actions/workflows/python-app.yml/badge.svg" alt="Pytest and Ruff validation">
      <img src="https://img.shields.io/coveralls/Quang00/qnscheduling.svg?logo=Coveralls" alt="Coverage Status">
      <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-ffdd54?logo=python&logoColor=ffdd54" alt="Python Version">
      <img src="https://img.shields.io/readthedocs/qnscheduling?logo=readthedocs&logoColor=white" alt="Documentation">
   </div>

----

**QNScheduling** is a simulation framework for scheduling quantum network applications using **Packet Generation Attempts (PGAs)**.
The simulator simulates on-demand entanglement packet scheduling with static and dynamic scheduling strategies, different routing schemes,
network-layer modeling, and stochastic entanglement generation with link contention.

.. raw:: html

   <div style="margin: 30px 0;"></div>

Overview
========

Key Features
------------

**Application Generation**
   Generate batches of quantum applications with configurable source/destination nodes, periods, packet counts, and EPR pair requirements.

**Budget Computation**
   Compute per-application PGA budgets based on network-layer models and entanglement swapping.

**Flexible Scheduling**
   Choose between **static EDF timetables** or **dynamic online EDF-like** scheduling strategies.

**Stochastic Simulation**
   Run simulations of entanglement generation/swapping with link contention, deferrals, retries, and drops.

**Comprehensive Analysis**
   Export results and summary metrics as CSV files for analysis.

Architecture
------------

Dynamic Online Scheduler Workflow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The high-level workflow of the dynamic scheduler for entanglement packets:

.. image:: dynamic_online_scheduler_pga.png
   :align: center
   :alt: Dynamic Online Scheduler
   :width: 700px

.. raw:: html

   <div style="margin: 20px 0;"></div>

Network-Layer Model
^^^^^^^^^^^^^^^^^^^

Example of the network-layer model with two schedulers (linear chain A-B-C) and 3 applications/flows:

.. image:: pga_detailed.png
   :align: center
   :alt: Network-Layer Model
   :width: 700px

.. raw:: html

   <div style="margin: 20px 0;"></div>


Network Topologies
------------------

QNScheduling provides basic and advanced network topologies stored as `.gml` files.

**Basic Topologies** include simple networks like chains, meshes, rings, stars, and dumbbells:

.. image:: star.png
   :align: center
   :alt: Star Topology
   :width: 400px

.. raw:: html

   <div style="margin: 20px 0;"></div>

**Advanced Topologies** include real-world network topologies from the Internet Topology Zoo:

.. image:: garr.png
   :align: center
   :alt: GARR Topology
   :width: 500px

Getting Started
===============

Installation
------------

1. **Create a Python Virtual Environment:**

.. code-block:: bash

   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install -U pip

2. **Install the requirements:**

.. code-block:: bash

   pip install -r requirements.txt

3. **Verify the installation:**

.. code-block:: bash

   pytest

Quick Start
-----------

Run the simulator with default settings:

.. code-block:: bash

   python -m scheduling.main

Usage Guide
===========

Scheduling Modes
----------------

Static Scheduler
^^^^^^^^^^^^^^^^

Use ``--scheduler static`` for offline precomputed EDF scheduling:

- Builds a static schedule over a hyperperiod horizon
- Uses a conflict graph for concurrency (applications sharing links cannot overlap)
- Performs feasibility checks:

  - Rejects applications with utilization U = duration/period > 1
  - Rejects schedules that would miss deadlines (end > deadline)

Dynamic Scheduler
^^^^^^^^^^^^^^^^^

Use ``--scheduler dynamic`` for online EDF-like scheduling:

- Online scheduling with dynamic arrival handling
- Arrivals are periodic by default; use ``--arrival-rate`` for Poisson process arrivals
- Supports admission control, scheduling, deferral, retry, and drop mechanisms

Routing Strategies
------------------

QNScheduling provides different routing strategies:

Shortest Path Routing (Default)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``--routing shortest`` (default) for shortest path routing:

- Computes shortest paths that meet minimum fidelity requirements
- Minimizes the number of hops within the maximum path length (L_max) derived from fidelity constraints
- Suitable for most basic use cases and simple network topologies
- Does not consider link congestion or capacity constraints
- Selects the first feasible path found

Yen Random Routing
^^^^^^^^^^^^^^^^^^

Use ``--routing random`` for randomized path selection among feasible paths:

- Uses Yen's K-shortest paths algorithm
- Randomly selects among all shortest paths that satisfy fidelity constraints
- Provides load balancing through randomization
- A baseline for distributed routing without global network state awareness
- Each application gets a uniformly random feasible path

Hub-Aware Routing
^^^^^^^^^^^^^^^^^

Use ``--routing degree`` for hub-aware routing that avoids high-degree nodes:

- Selects paths that minimize the maximum node degree among internal (non-endpoint) nodes
- Helps avoid congestion at network hubs and central nodes
- Distributes load away from highly connected nodes
- Among feasible paths, choose the shortest path

Capacity-Aware Routing
^^^^^^^^^^^^^^^^^^^^^^^

Use ``--routing capacity`` for capacity-aware routing with explicit utilization tracking:

- Tracks link utilization and excludes overloaded links from routing
- Routes traffic through less congested paths to balance network load
- Requires setting ``--capacity-threshold`` to define the maximum acceptable link utilization
- Computes PGA duration for each application and updates link capacities incrementally
- Applications are routed in order of their relative deadline (PGA duration / period)
- Helps prevent bottlenecks in heavily loaded networks

All routing respect the minimum fidelity threshold by computing a maximum path length (L_max) based
on the application's minimum fidelity requirement and the network's initial fidelity.
Paths exceeding this length are not considered.

PGA Status
^^^^^^^^^^

Status appearing in ``pga_results.csv``:

:completed: Generated required E2E EPR pairs within budget time (PGA)
:failed: Ran but didn't generate enough pairs by the end of PGA
:retry: Failed attempt that is rescheduled again (dynamic scheduler)
:defer: Could not start due to busy links, rescheduled to later (dynamic scheduler)
:drop: Cannot start/finish by deadline constraints (dynamic scheduler)

Command-Line Options
--------------------

Run ``python -m scheduling.main --help`` for the complete list of options.

Network Configuration
^^^^^^^^^^^^^^^^^^^^^

``--config``, ``-c``
   Path to network topology ``.gml`` file (default: ``configurations/network/Dumbbell.gml``)

``--routing``, ``-r``
   Routing scheme: ``shortest`` (default, fidelity-constrained shortest path), ``random`` (Yen random), ``degree`` (hub-aware), or ``capacity`` (capacity-aware)

``--capacity-threshold``, ``-ct``
   Capacity threshold for routing capacity per link (used with capacity routing)

Application Parameters
^^^^^^^^^^^^^^^^^^^^^^

``--apps``, ``-a``
   Number of applications to generate (a)

``--inst MIN MAX``, ``-i MIN MAX``
   Range of number of required entanglement packets per application (I_a)

``--epr MIN MAX``, ``-e MIN MAX``
   Range for EPR pairs requested per application (q_a)

``--period MIN MAX``, ``-p MIN MAX``
   Range for application periods in seconds (T_a)

``--min-fidelity MIN MAX``, ``-f MIN MAX``
   Range for application minimum fidelity (F_a). If omitted, minimum fidelity is not considered.

Physical Layer Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^

``--ppacket``, ``-pp``
   Target probability to compute PGA duration (p_packet)

``--memory``, ``-m``
   Memory multiplexing: number of independent link-generation trials per slot (m)

``--pswap``, ``-ps``
   Bell State Measurement probability success (p_bsm)

``--pgen``, ``-pg``
   EPR generation success probability per trial (p_gen)

``--slot-duration``, ``-sd``
   Slot duration in seconds (τ)

Scheduler Options
^^^^^^^^^^^^^^^^^

``--scheduler``, ``-sch``
   Scheduling strategy: ``static`` or ``dynamic``

``--hyperperiod``, ``-hp``
   Number of hyperperiod (H_i) cycles to schedule/simulate (static scheduler only)

``--arrival-rate``, ``-ar``
   Mean arrival rate λ for Poisson arrivals (dynamic scheduler). If omitted, arrivals are periodic.

General Options
^^^^^^^^^^^^^^^

``--seed``, ``-s``
   RNG seed for reproducibility (NumPy)

``--output``, ``-o``
   Output directory root (default: ``results``)

Example Commands
----------------

Dynamic Scheduler with Poisson Arrivals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m scheduling.main \
     --config configurations/network/basic/Dumbbell.gml \
     --apps 2 \
     --inst 2 2 \
     --epr 2 2 \
     --period 10.0 10.0 \
     --ppacket 0.1 \
     --memory 50 \
     --pswap 0.95 \
     --pgen 0.001 \
     --min-fidelity 0.6 0.6 \
     --slot-duration 0.0001 \
     --seed 42 \
     --scheduler dynamic \
     --arrival-rate 1.0 \
     --output results

Static Scheduler with Hyperperiod
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m scheduling.main \
     --config configurations/network/basic/Dumbbell.gml \
     --apps 2 \
     --inst 2 2 \
     --epr 2 2 \
     --period 10.0 10.0 \
     --hyperperiod 2 \
     --ppacket 0.1 \
     --memory 50 \
     --pswap 0.95 \
     --pgen 0.001 \
     --min-fidelity 0.6 0.6 \
     --slot-duration 0.0001 \
     --seed 42 \
     --scheduler static \
     --output results

Capacity-Aware Routing with Dynamic Scheduler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   python -m scheduling.main \
     --config configurations/network/basic/Dumbbell.gml \
     --apps 2 \
     --inst 2 2 \
     --epr 2 2 \
     --period 10.0 10.0 \
     --ppacket 0.1 \
     --memory 50 \
     --pswap 0.95 \
     --pgen 0.001 \
     --min-fidelity 0.6 0.6 \
     --slot-duration 0.0001 \
     --seed 42 \
     --routing capacity \
     --capacity-threshold 0.8 \
     --scheduler dynamic \
     --arrival-rate 1.0 \
     --output results

Output and Metrics
==================

Output Structure
----------------

Each run creates a new folder under ``results/seed_<seed>/runN/`` containing:

``app_requests.csv``
   Per-application request details including source node, destination node, minimum fidelity, period, and EPR requirements.

``link_utilization.csv``
   Per-link busy time and utilization statistics over the observed makespan.

``link_waiting.csv``
   Per-link waiting totals, average waiting time, and average queue length metrics.

``params.csv``
   Complete parameter set used for the run, including runtime and run number.

``pga_results.csv``
   Detailed per-attempt (PGA) logs with arrival time, start time, burst duration, completion time, waiting time, status, and more.

``summary.csv``
   Aggregate metrics including makespan, throughput, completion ratios, waiting statistics, and utilization statistics.

``summary_per_app.csv``
   Per-application breakdown showing counts of each status type and PGA duration statistics.

Key Performance Metrics
-----------------------

**Throughput Metrics**
   - Througput (PGAs completed per unit time)
   - Completion ratios (completed/total PGAs)

**Timing Metrics**
   - Total makespan (simulation time)
   - Average PGA duration
   - Per-link waiting times
   - P90/P95 waiting times

**Resource Utilization**
   - Per-link utilization
   - P90/P95 utilization
   - Per-link queue lengths
   - P90/P95 queue lengths

**Quality Metrics**
   - Average Minimum fidelity per application
   - Average number of hops
   - Drop ratios (due to deadline misses)
   - Retry and deferral statistics (dynamic scheduler)

API Documentation
=================

.. toctree::
   :maxdepth: 2
   :caption: Modules

   modules

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
