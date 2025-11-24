# 🧠 Machine Learning from Scratch

Welcome to **ML from Scratch**! This repository contains Python implementations of fundamental Machine Learning algorithms, built from the ground up. The goal is to understand the inner workings of these algorithms without relying on high-level abstractions like `scikit-learn` for the core logic.

These implementations rely primarily on **NumPy** for efficient vectorization and linear algebra operations.

## 🚀 Getting Started

### Dependencies
To run the code, you will need Python installed along with the following libraries:

```bash
pip install numpy matplotlib scikit-learn
```
*Note: `scikit-learn` is used **only** for generating synthetic datasets and splitting data for testing, not for the algorithms themselves.*

### Usage
Each algorithm is contained in its own directory. You can run the Python script directly to see a demonstration on a toy dataset:

```bash
# Example: Running Linear Regression
python "Linear Regression/linear_regression.py"
```

---

## 📚 Implemented Algorithms

### 1. Linear Regression
**Directory:** `Linear Regression/`
- **What it is:** A linear approach to modeling the relationship between a scalar response and one or more explanatory variables.
- **Implementation:**
  - Uses **Gradient Descent** to iteratively update weights and bias.
  - Minimizes the **Mean Squared Error (MSE)** cost function.
  - **Key Method:** `fit(X, y)` updates parameters using the learning rate and gradients.

### 2. Logistic Regression
**Directory:** `Logistic Regression/`
- **What it is:** A statistical model used for binary classification. It predicts the probability that a given input belongs to a certain class.
- **Implementation:**
  - Applies the **Sigmoid** activation function to map predictions to probabilities between 0 and 1.
  - Uses **Gradient Descent** to minimize the log-loss (cross-entropy).
  - **Key Method:** `_sigmoid(x)` transforms the linear output.

### 3. Decision Trees
**Directory:** `Decision Trees/`
- **What it is:** A flowchart-like structure where an internal node represents a feature, the branch represents a decision rule, and each leaf node represents the outcome.
- **Implementation:**
  - Recursively splits data based on the feature that provides the highest **Information Gain**.
  - Uses **Entropy** to measure impurity.
  - Supports `max_depth` and `min_samples_split` to control tree growth and prevent overfitting.

### 4. Random Forest
**Directory:** `Random Forest/`
- **What it is:** An ensemble learning method that constructs a multitude of decision trees at training time.
- **Implementation:**
  - Builds multiple `DecisionTree` instances.
  - Uses **Bootstrap Aggregation (Bagging)** to train each tree on a random subset of samples.
  - Aggregates predictions via **Majority Voting**.

### 5. Naive Bayes
**Directory:** `Naive Bayes/`
- **What it is:** A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
- **Implementation:**
  - Implements **Gaussian Naive Bayes**, assuming continuous features follow a normal distribution.
  - Calculates class **priors**, **means**, and **variances** during training.
  - Uses the Gaussian Probability Density Function (PDF) to compute posterior probabilities.

### 6. K-Nearest Neighbors (KNN)
**Directory:** `KNN/`
- **What it is:** A non-parametric method used for classification and regression.
- **Implementation:**
  - **Lazy Learner:** Simply stores the training data.
  - Calculates **Euclidean Distance** between the input and all training samples.
  - Returns the most common label among the `k` closest neighbors.

### 7. Support Vector Machines (SVM)
**Directory:** `Support Vector Machines/`
- **What it is:** A supervised learning model that analyzes data for classification and regression analysis.
- **Implementation:**
  - Linear SVM using **Hinge Loss**.
  - Optimizes weights using **Gradient Descent** with a regularization parameter (`lambda`).
  - Finds the hyperplane that maximizes the margin between classes.

### 8. Perceptron
**Directory:** `Perceptron/`
- **What it is:** The simplest type of feedforward neural network, a linear binary classifier.
- **Implementation:**
  - Uses the **Unit Step Function** (Heaviside step function) as the activation.
  - Updates weights only when a prediction is incorrect (Perceptron Learning Rule).
  - Visualizes the decision boundary in the test script.

---


