# AI Interpretability Journey: Find-Min

## Overview

This project is part of the **AI Interpretability Journey**, where the goal is to understand how transformers process information through simple and interpretable tasks. The **Find-Min Task** is designed to train a transformer model to predict the **minimum value** from a list of numbers.

Inspired by:
- **Callum McDougall's TakeMax Problem**
- **Neel Nanda's 200 Problems in Mechanistic Interpretability**

The model's simplicity (a single layer and single head) ensures ease of training and interpretability.

## Task Description

The task involves:
- Input: A list of numbers.
- Output: Predicting the minimum value in the list.

This basic problem provides an excellent opportunity to explore how transformer models encode and compute information.

## Repository Structure

Here’s an overview of the repository:

```
├── analyze.ipynb        # Notebook for analyzing and visualizing model behavior
├── dataset.py           # Script for generating datasets
├── model.py             # Transformer model implementation
├── training.py          # Script for training the model
├── training_model.ipynb # Notebook for training experiments
├── plotly_utils.py      # Utilities for generating visualizations
├── training_args.txt    # Training configurations
├── models/              # Directory for saved model checkpoints
├── images/              # Directory for visualization outputs
└── README.md            # Project documentation
```

## Features

- **Interpretable Transformers:** A small, single-head, single-layer transformer for simplicity.
- **Ablation Studies:** Analyze the model's attention mechanism by removing or modifying specific heads and positions.
- **Visualizations:** Tools to generate attention heatmaps and output vector visualizations.

## Getting Started

### Prerequisites

Ensure the following libraries are installed:

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
cd Transformer-Circuits/Find-Min
pip install -r requirements.txt
```

### Running the Notebooks

1. Train the model using `training_model.ipynb`.
2. Visualize the model’s behavior with `analyze.ipynb`.

## Key Functions

- **Prediction and Ablation:**
  - `predict_with_ablation`: Analyze how specific head or position modifications affect predictions.
- **Visualization:**
  - `calculate_qk_attn_heatmap`: Generate query-key attention heatmaps.
  - `plot_ov_heatmap_all_heads`: Visualize output vector weights across heads.

## Goals

1. Learn how transformers process information for simple tasks.
2. Develop intuition for attention mechanisms and weight interpretation.
3. Build a foundation for tackling more complex interpretability challenges.

## Acknowledgements

- **Callum McDougall** for the TakeMax Problem.
- **Neel Nanda** for the 200 Problems in Mechanistic Interpretability.

---

Dive into the fascinating world of interpretable AI and uncover how transformers make decisions!
