# ⚽ International Football Match Result Prediction

An end-to-end Machine Learning application that predicts international football match outcomes (`Home Win`, `Away Win`, or `Draw`) based on team match statistics using historical data from 1992 to 2026.

---


## 🔗 Live Demo & Links
* 🚀 **Interactive Streamlit App:** [Click Here to View Live App](https://football-prediction-app.streamlit.app/)
* 📁 **GitHub Repository:** https://github.com/ahmedhosary2020ahmed-sketch/Football_Prediction

## 📌 1. Problem Definition & Objective
The objective of this project is to build and deploy a machine learning model capable of predicting the outcome of an international football match. By analyzing historical match metadata and team metrics, the model evaluates whether a match will result in a **Home Win**, **Away Win**, or **Draw**.

---

## 📊 2. Dataset Selection
* **Dataset Name:** International Football Dataset from 1992 to 2026
* **Source:** Kaggle (`arrnalireza/international-football-dataset-from-2004-to-2026`)
* **Target Variable:** `result` — Categorical (`Home Win`, `Away Win`, `Draw`), generated dynamically by comparing `home_score` and `away_score`.
* **Features:** Historical match statistics, neutral ground status, and encoded match metadata.

---

## 📈 3. Exploratory Data Analysis (EDA)
Exploratory analysis was performed using `matplotlib` and `seaborn`:
* Analyzed match distribution across different tournaments (`sns.countplot`).
* Evaluated the impact of neutral venues (`neutral` feature) on match results.

---

## 🧹 4. Data Preprocessing & Pipeline
1. **Target Creation:** Calculated match outcomes into 3 distinct classes (`Home Win`, `Away Win`, `Draw`).
2. **Feature Dropping:** Dropped irrelevant or non-predictive columns (`Unnamed: 0`, `date`, `home_team`, `away_team`, `home_score`, `away_score`, `tournament`, `city`, `country`, `gd`, `home_res`).
3. **Missing Values:** Removed missing records via `df.dropna()`.
4. **Encoding:**
   * Applied `LabelEncoder` to categorical text columns.
   * Converted boolean values (`neutral`) to binary integers (`0` / `1`).
5. **Dataset Splitting:**
   * **Training Set:** 70%
   * **Validation Set:** 15%
   * **Testing Set:** 15%
   *(Stratified splits applied to preserve class balance)*

---

## 🤖 5. Model Development
* **Model Chosen:** `RandomForestClassifier` (`sklearn.ensemble`)
* **Hyperparameters:**
  * `n_estimators`: 300
  * `max_depth`: 12
  * `min_samples_leaf`: 20
  * `random_state`: 42
* **Reason for Selection:** Random Forest effectively captures non-linear relationships in sports metrics, reduces overfitting via ensemble averaging, and handles tabular feature interactions reliably.

---

## 📏 6. Model Evaluation
Evaluated performance using classification metrics:
* **Validation Accuracy:** Measured on the 15% validation split.
* **Test Accuracy:** Evaluated on the unseen 15% test split.
* **Evaluation Metrics:** Accuracy Score, Classification Report (Precision, Recall, F1-Score), and Confusion Matrix.

---

## 📂 7. Project Structure & Saved Artifacts
```text
├── app.py                            # Streamlit interactive application
├── football_model.pkl                # Serialized trained Random Forest model
├── defaults.pkl                      # Feature default values for user input
├── ranges.pkl                        # Valid feature min/max boundaries
├── requirements.txt                  # Application dependencies
└── README.md                         # Detailed project documentation
