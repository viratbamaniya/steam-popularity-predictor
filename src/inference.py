import pandas as pd
import numpy as np
import joblib

# Load
model = joblib.load('../models/random_forest_model.pkl')

def predict_popularity(price, positive, negative, playtime):
    X = pd.DataFrame([{
        'price': price,
        'positive_log': np.log1p(positive),
        'negative_log': np.log1p(negative),
        'playtime_log': np.log1p(playtime)
    }])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return pred, prob

if __name__ == "__main__":
    p, prob = predict_popularity(
        price=29.99,
        positive=150_000,
        negative=5_000,
        playtime=3500
    )
    print(f"Prediction (0=Not Popular, 1=Popular): {p}")
    print(f"Probability Popular: {prob:.4f}")
