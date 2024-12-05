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

Here’s the updated **Resources** section with the additional links included:

---

## 📚 Resources

Below is a collection of resources that have been instrumental in understanding transformer interpretability. These are grouped into **Videos**, **Blog Posts**, and **Papers/Tools** for easy navigation.

### **Videos**
1. [Mechanistic Interpretability: Attention Heads & Induction Heads](https://youtu.be/KV5gbOmHbjU?si=qBljml-mCcuDn1iU)  
   *(An intuitive introduction to attention heads and their role in transformers.)*

2. [Exploring Algorithmic Tasks in Transformers](https://youtu.be/bOYE6E8JrtU?si=vP7l66u_GeK0V_fP)  
   *(Discusses how transformers tackle algorithmic challenges.)*

3. [Visualizing Transformer Circuits](https://youtu.be/dsjUDacBw8o?si=cQgioyv_nvvae97L)  
   *(Detailed visualizations and explanations of transformer circuits.)*

---

### **Blog Posts**
1. [Getting Started with Mechanistic Interpretability by Neel Nanda](https://www.neelnanda.io/mechanistic-interpretability/getting-started)  
   *(A beginner-friendly introduction to mechanistic interpretability.)*

2. [Explaining the Transformer Circuits Framework by Example](https://www.lesswrong.com/posts/CJsxd8ofLjGFxkmAP/explaining-the-transformer-circuits-framework-by-example)  
   *(A practical example-driven explanation of transformer circuits.)*

3. [Induction Heads Illustrated](https://www.lesswrong.com/posts/TvrfY4c9eaGLeyDkE/induction-heads-illustrated)  
   *(A visual guide to understanding induction heads in transformers.)*

4. [September Monthly Problem: Mechanistic Interpretability](https://www.perfectlynormal.co.uk/blog-september-monthly-problem)  
   *(A hands-on problem to explore transformer circuits.)*

5. [200 COP in MI: Interpreting Algorithmic Problems](https://www.lesswrong.com/posts/ejtFsvyhRkMofKAFy/200-cop-in-mi-interpreting-algorithmic-problems)  
   *(A comprehensive blog exploring interpretability challenges in transformers.)*

6. [Measuring Structure Development in Algorithmic Transformers](https://www.alignmentforum.org/posts/ooAao2RYdFSd77qfp/measuring-structure-development-in-algorithmic-transformers)  
   *(Insights into how structure develops in transformers solving algorithmic tasks.)*

7. [Transformer Circuits: Main Hub](https://transformer-circuits.pub/)  
   *(A collection of articles exploring transformer circuits in depth.)*

8. [Dynalist: Interpretability Resources](https://dynalist.io/d/n2ZWtnoYHrU1s4vnFSAQ519J#z=aGu9fP1EG3hiVdq169cMOJId)  
    *(A curated list of resources for mechanistic interpretability.)*

---

### **Papers/Tools**
1. [ARENA 3.0: Mechanistic Interpretability Tools](https://github.com/callummcdougall/ARENA_3.0)  
   *(A GitHub repository providing tools and examples for mechanistic interpretability.)*

2. [Streamlit App: Exploring Chapter 1 Transformer Interpretability](https://arena3-chapter1-transformer-interp.streamlit.app/)  
   *(Interactive app for exploring transformer interpretability concepts.)*

3. [The Alignment Forum](https://www.alignmentforum.org/)  
   *(A hub for discussions and research on alignment and interpretability.)*

4. [Recent Paper: “Investigating Structure in Transformers”](https://arxiv.org/html/2311.04131v6)  
   *(A recent research paper on transformer interpretability and structure.)*

5. [Transformer Circuits Framework (by Anthropic)](https://transformer-circuits.pub/2021/framework/index.html)  
   *(A structured introduction to transformer circuits.)*

6. [In-Context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)  
   *(A deep dive into in-context learning and induction head mechanisms.)*
---

## 🚀 How to Use This Repository

- Navigate to individual project directories for detailed implementations and analysis.
- Run the Jupyter Notebooks (`analyze.ipynb`) to reproduce interpretability visualizations and insights.
- Use the `training.py` and `dataset.py` scripts to experiment with training new models on custom data.

---

Feel free to raise issues, provide feedback, or suggest improvements to the repository. Let's learn together!
