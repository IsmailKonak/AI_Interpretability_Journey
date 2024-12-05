# AI Interpretability Journey: Find-Median

## Overview

This repository is part of an **AI Interpretability Journey**, focused on exploring and demystifying the inner workings of transformer models through practical tasks. Specifically, this project tackles the **Find-Median Task**, a basic numerical task designed to make models both effective and interpretable.

The repository uses a minimalist approach: a single-layer, single-head transformer model. By simplifying the architecture, the project aims to shed light on how transformers encode and process information.

## Task Description

The **Find-Median Task** requires the model to predict the median value from a list of numbers. This simple yet insightful task is inspired by:
- **Callum McDougall's TakeMax Problem**
- **Neel Nanda's 200 Problems in Mechanistic Interpretability**

## Repository Structure

Here’s an overview of the repository’s contents:

```
├── analyze.ipynb        # Notebook for analyzing and visualizing model behavior
├── dataset.py           # Script for generating and managing datasets
├── model.py             # Transformer model implementation
├── training.py          # Model training script
├── training_model.ipynb # Notebook for training experiments
├── plotly_utils.py      # Utilities for visualizations
├── training_args.txt    # Training arguments and configuration
├── median_model4.pt     # Pretrained model checkpoint
└── README.md            # Project introduction and documentation
```

## Features

- **Transformers for Simple Tasks:** A transformer model designed to solve the median prediction problem.
- **Analysis and Interpretability:** Tools and utilities for understanding the model’s internals, such as attention heatmaps and weight visualizations.
- **Ablation Studies:** Support for experimenting with head and layer ablations to evaluate their impact on model performance.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:

- `torch`
- `numpy`
- `matplotlib`
- `plotly`
- `einops`
- `fancy_einsum`
- `transformer_lens`
- `circuitsvis`
- `IPython`

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/IsmailKonak/AI_Interpretability_Journey.git
cd Transformer-Circuits/Find-Median
pip install -r requirements.txt
```

### Running the Notebooks

1. Open `training_model.ipynb` to train the model.
2. Use `analyze.ipynb` to visualize and interpret the trained model.

## Goals

1. **Understand Transformers Better:** Learn how attention and weights contribute to task performance.
2. **Encourage Interpretability:** Show how small, interpretable models can yield insights into AI behavior.
3. **Bridge Theory and Practice:** Apply ideas from interpretability research to real tasks.

## Acknowledgements

Special thanks to:
- **Callum McDougall** for the TakeMax Problem inspiration.
- **Neel Nanda** for his 200 Problems in Mechanistic Interpretability.

---

Enjoy experimenting with interpretable AI and transforming insights into knowledge!


Note: Feel free to reach out if you notice any inaccuracies or have suggestions to improve this project—I'm here to learn and greatly appreciate constructive feedback! You can contact me at i_konak@hotmail.com or open an issue in the repository.

---
