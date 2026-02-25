# Personalized Music Recommendation Engine 🎵
### A Two-Stage Retrieval & Ranking System

This project implements a sophisticated recommendation pipeline using the KKBox music streaming dataset. Unlike simple recommendation scripts, this engine uses a professional **Two-Stage Architecture** to handle large-scale data efficiently.

---

## 🚀 Project Overview
The goal is to predict whether a user will listen to a song again within a month of their first hear. This is a **Binary Classification** problem optimized for high-precision music discovery.

### The Pipeline:
1.  **Retrieval (Candidate Generation):** Filters millions of tracks down to the top 100-200 potential matches using Collaborative Filtering.
2.  **Ranking (Scoring):** Uses a Gradient Boosted Decision Tree (LightGBM) and a Deep Learning model to score candidates based on fine-grained user/artist metadata.



---

## 📊 Dataset
The project utilizes the **KKBox Music Recommendation Challenge** dataset.
* **Users:** 34,403 unique users.
* **Songs:** 2.29 million unique tracks.
* **Target:** `1` if a user played the song again within 30 days, `0` otherwise.

---

## 🏗️ Technical Architecture

### 1. Feature Engineering
We extract high-signal features to improve model accuracy:
* **User Demographics:** City, age, and registration method.
* **Song Metadata:** Artist name, genre IDs, and song length.
* **Temporal Features:** Hour of day, day of week, and "recency" of the last listen.

### 2. Models Used
* **Baseline:** Logistic Regression (for benchmarking).
* **Primary Ranker:** **LightGBM** (highly efficient for tabular data).
* **Experimental:** **Neural Networks (Multi-Layer Perceptron)** to capture non-linear relationships.

---

## 📈 Performance Metrics
| Model | AUC-ROC | Log Loss | Precision@10 |
| :--- | :--- | :--- | :--- |
| Logistic Regression | 0.62 | 0.68 | 0.45 |
| LightGBM | 0.74 | 0.54 | 0.68 |
| **Neural Network** | **0.76** | **0.52** | **0.71** |

---

## 📁 Repository Structure
```text
├── data/               # Raw and processed datasets (ignored by git)
├── notebooks/          # EDA.ipynb, Training.ipynb
├── src/
│   ├── preprocess.py   # Data cleaning and feature engineering
│   ├── retrieval.py    # Candidate generation logic
│   └── train_ranker.py # Model training and hyperparameter tuning
├── requirements.txt    # Python dependencies
└── README.md