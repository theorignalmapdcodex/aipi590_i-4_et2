# Explainable AI: PDP, ICE, and ALE Plot Analysis

## Table of Contents
- [Explainable AI: PDP, ICE, and ALE Plot Analysis](#explainable-ai-pdp-ice-and-ale-plot-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Goal](#project-goal)
  - [Installation](#installation)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Key Features](#key-features)
  - [Analysis Components](#analysis-components)
  - [Results and Insights](#results-and-insights)
  - [Limitations and Considerations](#limitations-and-considerations)
  - [References](#references)

## Overview
This project implements and analyzes various explainable AI (XAI) techniques, specifically focusing on Partial Dependence Plots (PDP), Individual Conditional Expectation (ICE) plots, and Accumulated Local Effects (ALE) plots. These techniques are applied to a dataset containing user engagement metrics and conversion rates to uncover insights into user behavior and factors influencing conversions.

## Project Goal
The main objective is to gain a deep understanding of explainable AI techniques and their application in analyzing user engagement and conversion data. By utilizing PDP, ICE, and ALE plots, we aim to identify patterns, trends, and potential factors that influence user behavior and conversion likelihood.

## Installation
```bash
pip install numpy pandas scikit-learn shap imbalanced-learn
pip install git+https://github.com/MaximeJumelle/ALEPython.git@dev#egg=alepython
```

## Dependencies
- numpy
- pandas
- scikit-learn
- shap
- imbalanced-learn
- alepython
- matplotlib
- seaborn

## Usage
1. Clone the repository
2. Install the required dependencies
3. Run the Jupyter notebook `pdp+ice+ale_plot_interpretation.ipynb`

## Key Features
- Implementation of PDP, ICE, and ALE plots
- Analysis of user engagement metrics and their impact on conversion rates
- Visualization of feature importance and interactions
- Interpretation of model predictions and feature effects

## Analysis Components
1. Data Preprocessing:
   - Handling missing values
   - Feature engineering
   - Data normalization

2. Model Training:
   - Implementation of machine learning models (e.g., Random Forest, Gradient Boosting)

3. Explainable AI Techniques:
   - Partial Dependence Plots (PDP)
   - Individual Conditional Expectation (ICE) plots
   - Accumulated Local Effects (ALE) plots

4. Interpretation and Insights:
   - Analysis of feature importance
   - Identification of non-linear relationships
   - Understanding feature interactions

## Results and Insights
(Note: Specific results would be added after running the analysis)

## Limitations and Considerations
- The effectiveness of the plots may vary depending on the complexity of the model and the nature of the data
- Interpretation of the plots requires domain knowledge and careful consideration of potential confounding factors
- The analysis is limited to the features present in the dataset and may not capture all real-world factors influencing user behavior

## References
1. SHAP Documentation. [SHAP Text Examples](https://shap.readthedocs.io/en/latest/text_examples.html#question-answering)
2. Molnar, C. (2019). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*. [Online Book](https://christophm.github.io/interpretable-ml-book/)
3. Hugging Face. [Transformers Documentation](https://huggingface.co/docs/transformers/model_doc/bert)
4. DataCamp. [Explainable AI Tutorial](https://www.datacamp.com/tutorial/explainable-ai-understanding-and-trusting-machine-learning-models)
5. Hugging Face. [API Tokens](https://huggingface.co/settings/tokens)
6. Google. [Gemini AI](https://deepmind.google/technologies/gemini/)
   
---

📚 **Author of Notebook:** Michael Dankwah Agyeman-Prempeh [MEng. DTI '25]