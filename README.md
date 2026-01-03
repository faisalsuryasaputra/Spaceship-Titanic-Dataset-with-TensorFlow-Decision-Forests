# 🚀 Spaceship Titanic: TensorFlow Decision Forests

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.11%2B-orange?logo=tensorflow&logoColor=white)
![TF-DF](https://img.shields.io/badge/TF--DF-Random%20Forest-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

> **A robust baseline solution using TensorFlow Decision Forests (TF-DF) for the Kaggle Spaceship Titanic competition.**

## 📌 Overview

This project aims to predict whether a passenger on the Spaceship Titanic was transported to an alternate dimension. We utilize **TensorFlow Decision Forests (TF-DF)**, a library that enables the training of tree-based models (such as Random Forest) using the familiar and easy-to-use Keras API.

This approach is highly effective for tabular data as it requires minimal preprocessing compared to classical Neural Networks.

## 📊 Dataset

The dataset is sourced from the Kaggle competition: [Spaceship Titanic](https://www.kaggle.com/c/spaceship-titanic).

* **Total Training Data:** 8,693 entries with 14 features.
* **Target:** `Transported` (Boolean) - Whether the passenger was transported.
* **Key Features:** `HomePlanet`, `CryoSleep`, `Cabin`, `Destination`, `Age`, `VIP`, and luxury amenities expenditure (`RoomService`, `FoodCourt`, etc.).

## 🛠️ Methodology & Workflow

The code in this repository covers the following *end-to-end* steps:

### 1. Exploratory Data Analysis (EDA)
We analyzed the distribution of both numerical and categorical data to understand passenger characteristics and identify patterns.

### 2. Data Preprocessing
While TF-DF handles many data types natively, some adjustments were necessary:
* **Handling Missing Values:** Null values in numerical and boolean columns were imputed with `0`.
* **Boolean Conversion:** Since TF-DF does not currently support boolean data types directly, columns like `Transported`, `VIP`, and `CryoSleep` were converted to `integer` (0 or 1).
* **Dropping Columns:** Removed `PassengerId` and `Name` as they are irrelevant for training.

### 3. Feature Engineering
The `Cabin` feature, which contains data in the format `Deck/Num/Side`, was split into three more informative features:
* `Deck`
* `Cabin_num`
* `Side`

### 4. Modeling
We utilized the standard **Random Forest** algorithm from TF-DF.
* **Model:** `tfdf.keras.RandomForestModel()`
* **Data Split:** 80% Training, 20% Validation.
* **Data Format:** Converted Pandas DataFrame to `tf.data.Dataset` for optimal performance.

## 📈 Evaluation & Performance

The model was evaluated using accuracy metrics on the validation set and *Out-of-Bag* (OOB) data.

* **Training Time:** ~54 seconds.
* **OOB Accuracy:** ~79.73%
* **Validation Accuracy:** **80.25%**

### Feature Importance
Based on the `NUM_AS_ROOT` metric (how often a feature appears as the root of a tree), the most influential features are:
1.  **CryoSleep** (Highly dominant)
2.  **RoomService**
3.  **Spa**
4.  **VRDeck**

## 💻 How to Run

1.  **Install Dependencies:**
    ```bash
    pip install tensorflow tensorflow_decision_forests pandas numpy seaborn matplotlib
    ```

2.  **Run the Notebook:**
    Open the notebook file (e.g., `spaceship_titanic_tfdf.ipynb`) in Jupyter Notebook, Google Colab, or a Kaggle Kernel.

3.  **Output:**
    The script will generate a `submission.csv` file ready for upload to Kaggle.

## 🤝 Contribution
This repository is designed as a learning *baseline*. Feel free to fork and experiment with:
* Using `GradientBoostedTreesModel` instead of Random Forest.
* Conducting deeper hyperparameter tuning.
* Implementing more advanced missing data imputation techniques.

---
*Created based on TensorFlow Decision Forests v1.2.0 implementation.*
