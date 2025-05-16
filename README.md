# 📚 Hybrid Recommendation System (XGBoost + SVD + Ridge Regression)

This project implements a **hybrid rating prediction system** for the Yelp dataset, combining structured feature learning via **XGBoost** and collaborative filtering via **SVD**, with final predictions blended using **Ridge Regression**.

---

## 📌 Overview

The system is designed to improve RMSE by combining:
- 🔍 **Feature-based learning** using XGBoost on structured user-business interaction data
- 🎯 **Collaborative filtering** using SVD (Singular Value Decomposition)
- ⚖️ **Meta-model blending** using Ridge Regression to learn the optimal combination of predictions

📉 **Final RMSE**: `0.9776`  
⏱ **Runtime**: ~140 seconds

---

## 🧠 Feature Engineering

### 👤 User Features
- `review_count`, `useful`, `fans`, `average_stars`, `funny`, `cool`
- `elite_years`, `friend_count`, `compliment_sum`

### 🏪 Business Features
- `stars`, `review_count`, `RestaurantsPriceRange2`, `is_open`
- `HasTV`, `RestaurantsTakeOut`, `OutdoorSeating`, `WiFi`
- `checkin_count`, `photo_count`, `tip_count`, `avg_tip_length`

### 🔄 Interaction Features
- `user_avg_stars_diff`, `review_count_diff`, `compliment_to_popularity`
- `engagement_vs_visibility`, `high_fan_high_star`, `elite_friend_star_diff`
- `biz_star_minus_user_avg`, `user_review_count_ratio`, `biz_popularity`

---

## ⚙️ Models Used

### 1. **XGBoost Regressor**
- Learns from engineered features
- Tuned parameters:  
  `max_depth=5`, `learning_rate=0.03`, `n_estimators=1000`,  
  `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=5.0`, `reg_alpha=2.0`

### 2. **SVD (Surprise)**
- Captures latent user-item interactions
- Parameters: `n_factors=50`, `n_epochs=20`, `lr_all=0.005`, `reg_all=0.1`

### 3. **Ridge Regression (Meta-model)**
- Blends predictions from XGBoost and SVD

---

## 📈 Performance

### 🔍 Prediction Error Breakdown

| Error Range     | Count     |
|------------------|-----------|
| ≥0 and <1        | 102,428   |
| ≥1 and <2        | 32,635    |
| ≥2 and <3        | 6,145     |
| ≥3 and <4        | 835       |
| ≥4               | 1         |

