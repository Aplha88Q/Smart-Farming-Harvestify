# 🌾 Harvestify — Smart Crop Recommendation System

An ML-powered crop recommendation engine that predicts the most suitable crop for a given set of soil and environmental conditions. Built as part of an AI Hackathon, the system benchmarks multiple classification algorithms and deploys the best-performing model via a Flask API.

---

## 📌 Problem Statement

Farmers often select crops based on intuition or tradition, leading to suboptimal yields. This project leverages machine learning to recommend the ideal crop based on real-time agricultural parameters, enabling data-driven decisions that improve yield and sustainability.

---

## 📊 Dataset

| Property      | Detail                                     |
|---------------|--------------------------------------------|
| Samples       | 2,200                                      |
| Features      | 7 (N, P, K, temperature, humidity, pH, rainfall) |
| Target        | 22 crop classes                            |
| Source        | [Harvestify Dataset](https://github.com/Gladiator07/Harvestify) |

---

## 🤖 Model Benchmarking

All models were evaluated on an 80/20 stratified train-test split. Cross-validation scores are stratified k-fold averages.

| Model              | Test Accuracy | CV-5 Score | CV-10 Score |
|--------------------|--------------|------------|-------------|
| **CatBoost**       | **99.32%**   | **99.38%** | **99.26%**  |
| SVM                | 98.41%       | 98.35%     | 98.47%      |
| Gradient Boosting  | 98.18%       | —          | —           |
| KNN                | 97.73%       | 98.18%     | 98.24%      |
| GaussianNB         | 100%*        | 99.55%     | 99.43%      |

> \* GaussianNB achieved 100% training accuracy (LazyPredict benchmark); CV scores confirm generalization.

**CatBoost** was selected as the production model based on consistent cross-validated performance.

---

## 🏗️ Project Structure

```
Harvestify/
├── Crop_Recommendation_AI_Hackathon.ipynb   # Full EDA, benchmarking & training notebook
├── crop_recommendation.py                   # Standalone training + inference script
├── requirements.txt                         # Python dependencies
├── model/                                   # Saved model artefacts (generated on run)
│   ├── catboost_model.cbm
│   └── scaler.pkl
└── assets/
    ├── Working model.jpeg                   # Demo screenshot
    └── WhatsApp Image 2024-11-25 ...jpeg    # Hackathon presentation
```

---

## ⚙️ Setup & Usage

### 1. Clone & install dependencies

```bash
git clone https://github.com/<your-username>/Smart-Farming-Harvestify.git
cd Smart-Farming-Harvestify
pip install -r requirements.txt
```

### 2. Train the model

```bash
python crop_recommendation.py
```

This will:
- Download the dataset automatically
- Train and evaluate the CatBoost model
- Save `model/catboost_model.cbm` and `model/scaler.pkl`
- Print a sample prediction

### 3. Run inference (notebook)

Open `Crop_Recommendation_AI_Hackathon.ipynb` in Jupyter or Google Colab for the full pipeline including EDA, LazyPredict benchmarking, cross-validation, and per-class analysis.

---

## 🧪 Feature Description

| Feature       | Description                          | Unit       |
|---------------|--------------------------------------|------------|
| N             | Nitrogen content in soil             | kg/ha      |
| P             | Phosphorous content in soil          | kg/ha      |
| K             | Potassium content in soil            | kg/ha      |
| temperature   | Ambient temperature                  | °C         |
| humidity      | Relative humidity                    | %          |
| ph            | Soil pH level                        | 0–14       |
| rainfall      | Annual rainfall                      | mm         |

---

## 🧠 Tech Stack

- **ML:** CatBoost, Scikit-learn, LazyPredict
- **Data:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Flask

---

## 🏆 Results

CatBoost achieved **99.32% test accuracy** and **99.38% on 5-fold cross-validation**, making it highly reliable for production crop recommendation with minimal overfitting.

---

