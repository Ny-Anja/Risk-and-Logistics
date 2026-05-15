# Recharging Infrastructure Design for Robots Surveying Antarctica

**University of Edinburgh – School of Mathematics**
**Risk and Logistics – Assignment 2**

## Authors

| Name |
|---|
| Anja Harisoa |
| Mohamed Ali |
| Samwel Mlabwa |
| Selina Liyengwa |

---

## Overview

This project designs an optimal recharging network for a fleet of **1,072 battery-powered surveying robots** operating in Antarctica. Each robot must travel to a charging station either autonomously by drone (within battery range) or via a human-operated recovery vehicle (at a flat cost). The goal is to minimize total annualized cost — balancing station build costs, charger maintenance, drone charging costs, and human transport fees — by strategically placing stations and allocating chargers.

---

## Problem Structure

- **Fleet size:** 1,072 robots scattered across Antarctica
- **Transport modes:** autonomous drone flight (if within battery range `r_i`) or human vehicle (flat cost £1,000)
- **Station constraints:** max 8 chargers per station, max 2 robots per charger (capacity of 16 robots/station)
- **Cost components:** build (£5,000/station), maintenance (£500/charger), charging (£0.42/km), human transport (£1,000/robot)

---

## Methodology

### 1. MINLP Formulation
A **Mixed-Integer Non-Linear Programme** with continuous station coordinates as decision variables. Validated on small instances (20–60 robots) using the Xpress solver in Python. Achieves <3% optimality gap for instances up to 30 robots, but becomes computationally intractable beyond 60 robots within a 500-second time limit.

### 2. Construction Heuristic (K-Means)
A greedy construction heuristic using **K-Means clustering** to seed candidate station locations, followed by nearest-available-capacity assignment. Scales to the full 1,072-robot instance in under 1 second. Best results achieved with scaling factor γ = 2.

### 3. Local Search Improvement
A **location–allocation local search** with three neighbourhood moves:
- **N1 – Station Relocation:** moves each station to the geometric median of its assigned robots
- **N2 – Robot Re-allocation:** reassigns all robots to minimize individual cost given updated locations
- **N3 – Station Insertion:** opens a new station near the largest cluster of human-transport robots if cost-reducing

Achieves up to **18.1% improvement** over the construction heuristic, with GapUB of ~3% against the MINLP for instances up to 60 robots. Reduces the full 1,072-robot cost from £1,437,222 to **£1,305,629** (9.16% saving).

### 4. Stochastic Extension (SAA)
Extends the model to handle **uncertain battery ranges** across 100 scenarios using a **Sample Average Approximation (SAA)** two-stage stochastic framework:
- **Stage 1 (here-and-now):** station locations, charger counts, open indicators — fixed before scenarios are revealed
- **Stage 2 (wait-and-see):** robot assignments resolved per scenario

Evaluated with an 80/20 training/test split. The stochastic solution opens only **38 stations** (vs. 72 deterministic) and achieves a **Value of Stochastic Solution (VSS) of £292,954 (26.44%)**, generalizing well out-of-sample (26.62% on test scenarios).

---

## Key Results

| Method | Fleet Size | Cost (£) | Stations | Time |
|---|---|---|---|---|
| MINLP | 20–60 | 32,111 – 89,400 | 2–4 | up to 500s |
| Construction Heuristic (γ=2) | 1,072 | 1,437,222 | 72 | 0.20s |
| Local Search | 1,072 | 1,305,629 | 72 | ~33 min |
| Stochastic RP | 1,072 | 815,183 | 38 | ~43 min |

---

## Requirements

```
python
xpress
numpy
scikit-learn
scipy
matplotlib
pandas
```

---

## References

Ljósheim, H., Jenkins, S., Searle, K., and Wolff, J. (2026). Optimal placement of electric vehicle slow-charging stations: A continuous facility location problem under uncertainty. *Computers & Operations Research*, 185:107289.
