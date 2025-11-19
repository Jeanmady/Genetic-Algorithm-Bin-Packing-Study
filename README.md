# Genetic Algorithm (GA) Parameter Study for Bin Packing Problem (BPP)

This repository contains a self-contained Python script (`ga_bpp_experiment.py`) for an experimental study on a Genetic Algorithm applied to the Bin Packing Problem (BPP).

The detailed theoretical background, algorithm explanation, and in-depth analysis of the results are provided in the separate **Literature Review** document.

## Overview

The script implements a Genetic Algorithm designed to minimize the difference ($d$) between the heaviest and lightest bin weight for two BPP instances:

1.  **BPP1:** $K=500$ items, $B=10$ bins, **linear weights** ($w_i = i$).
2.  **BPP2:** $K=500$ items, $B=50$ bins, **quadratic weights** ($w_i = i^2 / 2$).

### Experiment Focus

The experiment is a parameter study focused on evaluating the performance of the GA under different configurations:

*   **Main Study:** Testing four combinations of **Mutation Rate ($\text{pm}$)** and **Tournament Size ($\text{t}$)** across BPP1 and BPP2.
*   **Further Experiments (BPP1 Only):** Investigating the influence of **Population Size ($\text{p}$)** and **Crossover Rate ($\text{pc}$)**.

The GA runs until **10,000 fitness evaluations** are completed for each trial, using a default Population Size $P=100$ and Crossover Rate $\text{pc}=0.8$. Each configuration is tested over **5 trials**.

## Running the Experiment

### Prerequisites

To install and synchronize the required dependencies using **uv**:

```bash
uv sync
```

## Outputs

The script will produce two main types of output:

Terminal Output: Prints the mean, standard deviation, and best performance (lowest d) for all parameter settings tested in the main and further experiments.

Generated Files: Two image files are saved to the current directory, showing the mean convergence curves (Best Fitness vs. Evaluations) for the 4 primary settings:

Figure1_BPP1_Convergence.png

Figure2_BPP2_Convergence.png

## Algorithm Details (Refer to Literature Review)

For a full breakdown of the implementation, including the Chromosome Representation, Fitness Function, Selection, Crossover, Mutation, and Replacement strategies, please refer to the accompanying Literature Review document.

## Credits

Made by Jean Mady :)