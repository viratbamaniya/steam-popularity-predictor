# 🎮 Steam Game Popularity Prediction

This project predicts Steam game popularity using a Random Forest ML model trained on reviews, price, and playtime.

📌 **Goal:** Build a machine learning model that predicts popularity so developers or publishers can get insights about their game’s potential.

---

## 📂 Data Source

The dataset comes from **[SteamSpy](https://steamspy.com/api.php)** — a third-party service that estimates game stats using Steam’s public API.

**Features used:**
- `price`: Price of the game (USD)
- `positive`: Number of positive reviews
- `negative`: Number of negative reviews
- `average_forever`: Average playtime in minutes
- `owners`: Estimated owner range (processed)

**Target variable:**  
- `popular`: Binary label → 1 if estimated owners ≥ 10 million, else 0

---

## 📊 Exploratory Data Analysis (EDA)

Key EDA steps:
- Checked distributions, boxplots, scatterplots for price, reviews, owners.
- Created `owners_est` as numeric lower bound from `owners` range.
- Applied `log` transforms to reviews and playtime for better scale handling.
- Labeled games as **Popular** if they have at least 10 million estimated owners.

---

## ⚙️ Modeling

I compared:
- ✅ **Logistic Regression** (linear baseline)
- ✅ **Decision Tree** (simple tree)
- ✅ **Random Forest** (final choice)

**Final Model:**  
**Random Forest Classifier** with ~95% accuracy.  
Selected for good performance, interpretability (feature importances), and no need for scaling.

**Most important features:**
1. Positive reviews
2. Negative reviews
3. Average playtime
4. Price

---

## 🚀 How to Use

Clone the repo, install dependencies (`requirements.txt`), then test predictions:

```bash
python src/inference.py
