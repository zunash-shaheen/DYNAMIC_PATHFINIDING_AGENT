# DYNAMIC_PATHFINIDING_AGENT
A grid-based **Dynamic Pathfinding Agent** implementing **A\*** and **Greedy Best-First Search (GBFS)** with real-time obstacle spawning and live re-planning — built entirely with **Pygame** .

---

## Features

| Feature | Details |
|---|---|
| **Algorithms** | A\* `f(n) = g(n) + h(n)` and GBFS `f(n) = h(n)` |
| **Heuristics** | Manhattan Distance and Euclidean Distance |
| **Grid Sizing** | User-defined rows x cols at startup (5-30) |
| **Map Editor** | Left-click / drag to place or remove walls |
| **Start & Goal** | Moveable via Click Mode buttons |
| **Random Maze** | Configurable density 5%-70% |
| **Dynamic Mode** | Obstacles spawn mid-run; agent re-plans instantly |
| **Visualisation** | Frontier (Yellow), Visited (Pink), Path (Magenta) |
| **Metrics** | Nodes Visited, Path Cost, Execution Time (ms), Re-plans |

---

## Installation

### Step 1 - Check your Python version
```bash
python --version
```

### Step 2 - Install pygame

| Python version | Command |
|---|---|
| 3.8 to 3.12 | `pip install pygame` |
| 3.13 or newer | `pip install pygame-ce` |

### Step 3 - Clone and run

```bash
git clone https://github.com/YOUR_USERNAME/dynamic-pathfinding-agent.git
cd dynamic-pathfinding-agent
pip install pygame
python pathfinding_agent.py
```

---

## Usage

When you run the program it will ask for grid size in the terminal:

```
====================================================
   Dynamic Pathfinding Agent  -  Pygame
====================================================
  Rows   (5-30, press Enter for 15):
  Cols   (5-30, press Enter for 20):
```

Press **Enter** to use the defaults (15 rows x 20 columns).

---

## GUI Controls

### Right Panel

| Control | Action |
|---|---|
| A* / GBFS radio | Select search algorithm |
| Manhattan / Euclidean radio | Select heuristic function |
| Wall / Start / Goal radio | Set what grid click does |
| - and + buttons | Adjust obstacle density |
| Random Maze | Fill grid with random walls |
| Clear Walls | Remove all walls |
| Run Search | Start the animation |
| Dynamic Mode OFF/ON | Toggle obstacle spawning |
| Reset View | Clear search visualisation |

### Grid Interaction

| Action | Effect |
|---|---|
| Left-click (Wall mode) | Toggle wall on/off |
| Click and drag (Wall mode) | Draw walls |
| Click (Start mode) | Move Start (S) node |
| Click (Goal mode) | Move Goal (T) node |
| ESC | Quit |

---

## How Dynamic Mode Works

1. Toggle **Dynamic Mode: ON**
2. Click **Run Search**
3. Agent instantly plans a path and begins moving step-by-step
4. Each step a random wall may spawn on a cell NOT on the current path (~18% chance)
5. If the new wall blocks future path -> **instant re-plan** from current position
6. If the wall does not affect the path -> agent keeps moving (no wasted re-plan)
7. The Re-plans counter in the metrics panel tracks how many times this happens

---

## Algorithm Summary

### A* Search
```
f(n) = g(n) + h(n)
```
- g(n): actual cost from start to n
- h(n): heuristic estimate from n to goal
- Optimal: always finds shortest path
- Complete: always finds a path if one exists

### Greedy Best-First Search (GBFS)
```
f(n) = h(n)
```
- Only uses heuristic, ignores actual cost
- Faster than A* in open environments
- Not optimal: may return a longer path

### Heuristics

| Heuristic | Formula | Best for |
|---|---|---|
| Manhattan | abs(r1-r2) + abs(c1-c2) | 4-directional grid |
| Euclidean | sqrt((r1-r2)^2 + (c1-c2)^2) | Open spaces |

---


## Dependencies

Only one external package required:

```
pygame >= 2.0
```

 Pure Python + Pygame only.

## Course Information

**Course:** Artificial Intelligence
**Assignment:** 2 - Dynamic Pathfinding Agent
**University:** National University of Computer & Emerging Sciences
