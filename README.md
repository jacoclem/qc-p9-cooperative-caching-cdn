# Quantum Computing Project: Cooperative Caching in CDN (P09)

This repository contains the implementation of **Project 9: Cooperative Caching in a Content Delivery Network**, developed for the Quantum Computing course (A.Y. 2026/2027) at Roma Tre University.

## Problem Description
Video streaming and download services distribute content through a Content Delivery Network (CDN). Edge servers, located close to users, have limited memory capacity and can only cache a subset of the available content. 

The goal is to determine which contents to store in the cache of each edge server to **maximize the overall benefit** (wait time saved by users). The problem includes:
- **Redundancy Penalties:** If neighboring servers store the same content, the marginal benefit decreases.
- **Capacity Constraints:** The sum of the sizes of the stored contents cannot exceed the server's memory capacity.
- **Replication Constraints:** To ensure fault tolerance, each content must be replicated across a minimum number of edge servers.

The problem is modeled as a **Constrained Binary Quadratic Model (BQM)** and solved using both Classical Exact/Heuristic Solvers and the **D-Wave Quantum Processing Unit (QPU)**.

## Repository Structure
- `data/`: Contains the JSON datasets (`data.json`) and the exported metrics.
- `src/`: 
  - `parser.py`: Parses the JSON instances and exposes a clean `CachingInstance` dataclass.
  - `bqm_builder.py`: Maps the objective function and constraints into a QUBO/BQM formulation using Lagrange multipliers.
  - `validator.py`: Evaluates solutions against physical constraints (memory and replicas) and computes the exact objective value.
  - `ortools_solver.py`: Classical exact baseline using Google OR-Tools (CP-SAT) with per-block statistics.
- `main.py`: The main orchestrator and CLI entry point.
- `docker-compose.yml` / `Dockerfile`: Containerized environment setup.
- `requirements.txt`: Python dependencies (Ocean SDK, OR-Tools, etc.).

## Prerequisites
The project is fully containerized. You only need:
- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)

## Setup and Execution

**1. Start the environment**
Build and run the Docker container to ensure all dependencies (including D-Wave Ocean SDK) are properly installed:
```bash
docker compose up -d --build
