# Multi-Agent PAC-MEN Challenge
- A Decentralized Negotiation and Resource Contention Simulation using Multi-Agent Systems

---

#Overview

This project implements a "Multi-Agent System (MAS)" based on the "Pac-Man" environment, where three autonomous agents (Pac-Men) compete and cooperate to collect pellets within a maze.  
The key challenge lies in "shared corridor conflicts", where agents must negotiate traversal rights through decentralized decision-making.

Two negotiation protocols are implemented and compared:

1. "Priority-Based Protocol (Baseline):"
   - Conflict resolution determined by agent score.
   - Deterministic and efficient, but potentially unfair.

2. "Alternating-Offer Protocol (Formal Negotiation):"
   - Agents exchange proposals and counteroffers until an agreement is reached or timeout occurs.
   - Promotes fairness through utility-based decision-making at the cost of added time complexity.

---

#Features

- Predictive Conflict Detection (1–2 step lookahead).
- Mutual Exclusion Locks for shared corridors.
- Local Sensing to detect nearby agent conflicts.
- Negotiation Protocol Switching: press keys "P" (Priority) or "O" (Alternating-Offer).
- Real-Time Logging to `negotiation_log.csv`.
- Dynamic Visualization via `pygame` GUI (maze, pellets, energy, fairness metrics).
- Ghost Agents with probabilistic pursuit and avoidance patterns.
- Automatic Summary Screen with fairness index, waiting times, success/failure counts.
- CSV Export for post-simulation analysis and figure generation.

---

## Repository Structure

multi-pacmen-simulation/
├── README.md
├── requirements.txt
├── src/
│   ├── multi_pacmen_final.py
├── data/
│   └── negotiation_log.csv
├── scripts/
│   └── analyze_results.py
└── results/
    └── figures/

---

#How to Run

# **1. Clone the Repository**
```bash
git clone https://github.com/irenecodes/multi-agent-pacmen.git
cd multi-agent-pacmen
