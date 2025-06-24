# Cooperative Matrix Game Environment for GOAT

Simple environment for experimenting with cooperative matrix game.

Author: Paresh R. Chaudhary (pareshrc@uw.edu)

## Overview

This repository provides a framework for creating and experimenting with cooperative game. It includes tools for generating game layouts, calculating payoffs, and simulating agent interactions.

## Project Structure

The project is organized into the following directories:

- `env/`: Contains the game environment implementations and matrix game logic
- `policy/`: Houses policy implementations for different agents
- `trainer/`: Contains training algorithms
- `utils/`: Utility functions and helper modules
- `runner/`: Scripts for running experiments and simulations
  - `goat.py`: Implementation of the GOAT algorithm
  - `goat_search.py`: Hyperparameter search routines for GOAT
  - `comedi.py`: Implementation of the comedi algorithm
  - `mep.py`: Implementation of the MEP algorithm
  - `trajedi.py`: Implementation of the trajedi algorithm
  - `sp.py`: Implementation of the SP (Self-Play) algorithm
- `results/`: Directory for storing experiment results and outputs

## Installation

```bash
# Clone the repository
git clone https://github.com/pareshrchaudhary/cooperative_matrix_game.git
cd cooperative_matrix_game

# Create and activate conda environment
conda create -n cmg python=3.11
conda activate cmg

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .

# Set the PYTHONPATH (required for imports to work correctly)
export PYTHONPATH=$PYTHONPATH:/path/to/parent/directory
# For example, if your package is at /Users/username/dev/cooperative_matrix_game:
export PYTHONPATH=$PYTHONPATH:/Users/username/dev
```

## Usage

After installation and setting the PYTHONPATH, you can run individual algorithm files to run experiments and generate results.:


```sh
python runner/mep.py
```

or

1. Generate CoMeDi data to train VAE.
```sh
python runner/comedi.py
```

2. Train VAE.
```sh
python trainer/vae_trainer.py
```

3. Train GOAT.
```sh
python runner/goat.py
```
or 

<a href="https://colab.research.google.com/github/pareshrchaudhary/cooperative_matrix_game/blob/main/GOAT.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Cite

```
@article{chaudhary2025improving,
  title={Improving Human-AI Coordination through Adversarial Training and Generative Models},
  author={Chaudhary, Paresh and Liang, Yancheng and Chen, Daphne and Du, Simon S and Jaques, Natasha},
  journal={arXiv preprint arXiv:2504.15457},
  year={2025}
}
```
