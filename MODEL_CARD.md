# Model Card: Salary Prediction via Linear Regression

## 1. Model Details
- **Model Name:** Linear Regression Salary Predictor
- **Version:** 1.0.0
- **Type:** Supervised Learning (Univariate Regression)
- **Framework:** `scikit-learn` Pipeline (`SimpleImputer(median)` -> `LinearRegression()`)
- **Model Serialization:** Joblib (`models/salary_model.joblib`)
- **Metadata:** `models/model_metadata.json`

## 2. Intended Use
- **Primary Use Case:** Estimating annual base compensation (USD) based on years of professional experience.
- **Target Audience:** HR compensation analysts, recruiters, hiring managers, and candidates seeking an empirical baseline for negotiation.
- **Out-of-Scope Uses:** 
  - Making binding automated compensation decisions without human review.
  - Applying without local cost-of-living adjustments across disparate international geographies.
  - Applying to executive levels (C-suite, Partners) or equity-heavy compensation packages.

## 3. Data & Features
- **Dataset:** `Salary_dataset.csv` (100 verified and expanded industry observations)
- **Feature Matrix ($X$):**
  - `YearsExperience` (Float): Total years of relevant professional experience ($0.8 \le \text{years} \le 14.8$).
- **Target Variable ($y$):**
  - `Salary` (Float): Annual compensation in USD ($26,211.00 \le \text{salary} \le \$166,633.00$).
- **Splitting Strategy:** 80% Training ($N=80$), 20% Testing ($N=20$) split using random seed `42` **before** fitting any preprocessing transformations to strictly eliminate data leakage.

## 4. Fitted Model Equation & Parameters
$$\text{Salary} = \beta_0 + \beta_1 \cdot \text{YearsExperience}$$
$$\text{Salary} \approx \$25,553.03 + (\$9,380.90 \times \text{YearsExperience})$$

- **Intercept ($\beta_0$):** $\$25,553.03$ (estimated starting baseline compensation for zero years experience).
- **Slope ($\beta_1$):** $\$9,380.90$ (expected marginal salary increase per additional year of experience).

---

## 5. Performance & Evaluation Metrics

| Metric | Training Set (80%) | Testing Set (20%) | Baseline (Dummy Mean) | Improvement vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Mean Absolute Error (MAE)** | **$3,940.48** | **$3,969.31** | $35,005.46 | **-88.7%** |
| **Root Mean Squared Error (RMSE)**| **$5,050.15** | **$4,944.93** | $37,809.20 | **-86.9%** |
| **Coefficient of Determination ($R^2$)** | **0.9832** | **0.9816** | -0.0739 | **+1.055 pts** |
| **Mean Absolute Percentage Error (MAPE)** | **5.75%** | **5.96%** | N/A | **High Precision** |

### Overfitting & Generalization Analysis
- The delta between Training $R^2$ ($0.9832$) and Testing $R^2$ ($0.9816$) is only **$0.0016$**.
- This indicates near-perfect generalization to unseen test data with no detectable overfitting.

---

## 6. Error Breakdown across Target Salary Tiers

| Salary Bracket | Test Count | Mean Actual Salary | Mean Predicted Salary | Bracket MAE | Bracket RMSE | Bracket MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Entry Level (<$60,000)** | 8 | $46,147.00 | $48,653.51 | $3,329.56 | $4,181.23 | 7.98% |
| **Mid Level ($60,000 - $100,000)** | 5 | $71,451.00 | $72,457.55 | $4,122.36 | $5,429.92 | 5.87% |
| **Senior Level (>$100,000)** | 7 | $126,276.86 | $124,856.61 | $4,591.12 | $5,368.76 | 3.73% |

### Key Diagnostic Takeaways:
1. **Percentage Error Decreases with Seniority:** While the absolute dollar error increases slightly in higher brackets ($3,329$ vs $4,591$), the percentage error (MAPE) steadily drops from **7.98%** in Entry Level to **3.73%** in Senior roles.
2. **Homoscedasticity:** Residuals exhibit fairly uniform variance across predicted values without severe heteroscedastic fan-shaped dispersion.
3. **Residual Normality:** Test residuals average close to zero ($-757.15$) with a standard deviation of $\$5,013.57$, closely matching the overall RMSE ($4,944.93$).

---

## 7. Limitations & Caveats

1. **Univariate Limitation:**
   - The model relies solely on `YearsExperience`. In reality, compensation is strongly influenced by role title (e.g., Software Engineer vs Support), education credentials, specialized certifications, and company tier.
2. **Geographical & Cost-of-Living Variance:**
   - Salaries do not factor in location (e.g., San Francisco vs Austin vs Bangalore).
3. **Extrapolation Risk:**
   - **Negative or near-zero inputs:** The model should not be fed negative numbers.
   - **High experience bounds (>18 years):** Linear wage growth typically plateaus or shifts into non-linear equity compensation; extrapolating past 20 years will overestimate base salary without proper capping.
4. **Inflation & Temporal Drift:**
   - Compensation scales shift annually with macro inflation. The model requires periodic retraining with recent market survey data.

---

## 8. Deployment & API Usage
The trained model is exposed via the Django REST Framework API:
- Endpoint: `POST /api/predict/`
- Request: `{"years_experience": 5.5}`
- Response:
```json
{
  "years_experience": 5.5,
  "predicted_salary": 77147.98,
  "currency": "USD",
  "confidence_interval_95": {
    "lower": 67455.92,
    "upper": 86840.04
  },
  "model_version": "1.0.0"
}
```
