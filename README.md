# 🛒 Olist E-Commerce Customer Review Analytics & Risk Predictor

An advanced Machine Learning web application designed to predict customer review outcomes in real-time for Brazil's largest e-commerce marketplace (Olist). This dashboard helps operations and customer success teams preemptively catch bad experience triggers before they hit the platform.

🔗 https://olist-risk-predictor.streamlit.app/

---

## 💡 Key Features & Architecture
- **Multi-Page Navigation Matrix:** Organized into a Real-Time Predictor, ML Diagnostics, and an Executive Corporate ROI Dashboard.
- **Geospatial Logistics Distance:** Implements the `Haversine Formula` to dynamically compute cross-regional shipping distance using customer and merchant latitude/longitude coordinates.
- **Hybrid Decision Guard Logic:** Blends predictive probabilities from a Random Forest model with real-world deterministic business constraints (e.g., severe delays, exorbitant freight prices).
- **Proactive Risk Attribution:** Explains the exact logistical metrics driving a high "Bad Review Risk Score".

---

## 🧠 Model Validation Metrics
- **Algorithm:** Random Forest Classifier (Optimized via Hyperparameter Tuning to control overfitting)
- **Baseline Accuracy:** `87.4%`
- **ROC-AUC Score:** `0.912`
- **Key Determinant Parameters:** Estimated Delivery Days, Seller Historical Reputation, Product Description Length, and Freight-to-Price Premium.

---

## 🛠️ Tech Stack
- **Backend:** Python (`scikit-learn`, `Pandas`, `NumPy`)
- **Frontend / Visualization:** Streamlit, Tailwind-inspired Dark CSS injection, Plotly Express
- **Serialization:** Pickle Data Serialization Matrix

---

## 🚀 Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
   cd YOUR_REPO_NAME
