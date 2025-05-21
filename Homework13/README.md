# 🌳 Exercise 13: Decision Tree Regression (Machine Learning Example)

This project demonstrates a **machine learning method not covered in class** — the **Decision Tree Regressor** — using `scikit-learn`. It includes a working example, performance evaluation, and a clear explanation of the underlying principle.

---

## 🧠 What is Decision Tree Regression?

Decision Tree Regression is a **non-linear supervised learning algorithm** used for **predicting continuous values**. It works by:

- Splitting the dataset into smaller and smaller groups
- Creating **decision rules** based on input features
- Predicting the **average value** within each final group (leaf node)

This is similar to asking a series of "yes/no" questions to narrow down the best prediction.

---

## 📌 Project Objectives

- ✅ Use a **decision tree** to fit a regression model
- ✅ Understand how tree-based splits minimize prediction error
- ✅ Evaluate performance using **Mean Squared Error**
- ✅ Visualize predictions vs actual values

---

## 🧪 Dataset

This example uses **synthetic regression data** generated via `make_regression` with added noise.

- Features: 1 (for easy visualization)
- Samples: 100
- Noise: 15

---

## 🏗️ Model Configuration

| Component        | Setting               |
|------------------|------------------------|
| Model            | `DecisionTreeRegressor` |
| Depth            | `max_depth=3`         |
| Evaluation Metric | `Mean Squared Error`  |
| Library          | `scikit-learn`        |

---

## 🚀 How to Run

### 📦 Requirements

```bash
pip install scikit-learn matplotlib
