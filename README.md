# Perceval-Quest-Solution

The Perceval Quest Solution by Team Quantum Tree.

## One-page Report

[View the PDF](./one-page.pdf)

## Overview

This repository contains the solution for the Perceval Quest by Team Quantum Tree. It implements the GLASE architecture and a classical CNN baseline for partial MNIST classification. The solution uses a BosonSampler to simulate quantum circuits and supports evaluation on a QPU via Scaleway. The project includes scripts for training, plotting results, and QPU validation.

## Directory Structure

```
.
├── requirements.txt
├── README.md
└── src
    ├── train.py          # Training script with simulator.
    ├── plot_result.py    # Script to plot detailed training metrics.
    └── qpu_validation.py # Script to evaluate the trained model on a QPU session.
```

## Installation

Before running the scripts, install the required Python packages by executing:

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script with default hyperparameters or specify your own using command-line arguments. For example:

```bash
python src/train.py --m 20 --n 3 --batch_size 256 --lr 2e-3 --weight_decay 1e-3 --epochs 50 --label_smoothing 0.1
```

If no arguments are provided, the script will use the default values. Run `python src/train.py -h` for detailed arguments.

### Plotting Training Results

To visualize the training metrics after training both QNN and CNN, use the plotting script. By default, it uses `qnn.pkl` and `cnn.pkl` files. You can override these by providing your own file paths:

```bash
python src/plot_result.py --files qnn.pkl cnn.pkl
```

### QPU Validation

Evaluate the model using a QPU session with the `qpu_validation.py` script. You will need to provide your Scaleway credentials and configuration parameters. For example:

```bash
python src/qpu_validation.py --scw_project_id YOUR_PROJECT_ID --scw_token YOUR_TOKEN --filename qnn.pkl --platform qpu:ascella --m 12 --n 3 --fraction 0.2 --batch_size 2
```

Replace `YOUR_PROJECT_ID` and `YOUR_TOKEN` with your actual Scaleway credentials.

## Requirements

All required packages are listed in the [`requirements.txt`](requirements.txt) file. Make sure to install them before running any scripts.