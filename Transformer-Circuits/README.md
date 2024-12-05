# Transformer Circuits: An Interpretability Journey

Welcome to the **Transformer Circuits** repository! This repository serves as a collection of my studies and experiments aimed at understanding the inner workings of transformer models, particularly focusing on interpretability concepts. My goal is to gain hands-on experience by exploring various transformer circuit mechanisms and documenting my journey.

## 🌟 Objectives

In this repository, I explore:
- **QK Circuits**: How the query-key interactions affect attention and model behavior.
- **OV Circuits**: Understanding the transformations and compositions in output-value circuits.
- **Compositional Interpretability**: Investigating how circuits combine to produce meaningful behaviors.
- **Toy Transformer Models**: Interpreting simplified models trained on algorithmic tasks for a more intuitive grasp of transformers.

By studying these concepts, I aim to build an intuition for the mathematical and algorithmic underpinnings of transformer architectures.

---

## 📂 Projects Overview

### 1. **Find-Median**
This project demonstrates how a transformer learns to compute the **median** of a given sequence of numbers. It involves:
- **Data Generation**: Custom scripts to generate datasets for median computation.
- **Model Training**: Training a transformer model on the generated dataset.
- **Analysis**: Interpreting the transformer’s attention and weights to uncover how it solves the median-finding task.

Key Files:
- `dataset.py`: Generates data for training the model.
- `training.py`: Contains the training loop and hyperparameters.
- `analyze.ipynb`: A Jupyter Notebook for analyzing the trained model's behavior and visualizing attention patterns.

---

### 2. **Find-Min**
This project focuses on teaching a transformer to compute the **minimum** value in a sequence. Similar to the **Find-Median** project, it emphasizes:
- **Data Creation**: Crafting datasets tailored for the minimum-finding task.
- **Model Training**: Training a transformer to learn the algorithm.
- **Interpretability**: Analyzing and visualizing how the model processes inputs to determine the minimum.

Key Files:
- `dataset.py`: Creates sequences with their minimum values for model training.
- `training.py`: Implements the training process.
- `analyze.ipynb`: Includes interpretability-focused experiments, such as visualizing attention heads and layer contributions.

---

## 📚 Resources

The following resources were instrumental in developing these projects:

### **Videos**
- [Neel Nanda’s Tutorials](https://www.youtube.com/): A step-by-step guide to understanding transformer circuits.

### **Blog Posts**
- [Transformer Circuits by Anthropic](https://transformer-circuits.pub/): In-depth explorations of attention mechanisms and interpretability techniques.

### **Papers**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): The seminal transformer architecture paper.
- [A Mathematical Framework for Transformer Circuits](https://arxiv.org/abs/): An advanced exploration of circuit-level mechanisms.

---

## 🚀 How to Use This Repository

- Navigate to individual project directories for detailed implementations and analysis.
- Run the Jupyter Notebooks (`analyze.ipynb`) to reproduce interpretability visualizations and insights.
- Use the `training.py` and `dataset.py` scripts to experiment with training new models on custom data.

---

Feel free to raise issues, provide feedback, or suggest improvements to the repository. Let's learn together!
