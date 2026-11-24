# Predicting Urban PM2.5 from Satellite & Weather Data

Predicts daily PM2.5 air-pollution levels for **179 cities the model has never seen**, using Sentinel-5P satellite readings (CO, NO₂, HCHO, SO₂, O₃, aerosol index) and GFS weather forecasts. Data Analytics course team project by **Nour Akl** and **Ahmed Hesham**, based on the World Air Quality Index challenge.

## Approach

- **Exploratory analysis:** target skew and log transform, the COVID-19 lockdown effect on pollution, satellite-band missingness (CH4 dropped at 81% missing), feature correlations and per-city patterns
- **Feature engineering grounded in atmospheric chemistry:**
  - per-city lag features
  - source-type ratios (HCHO/CO, NO₂/CO, SO₂/CO)
  - wind magnitude and direction
  - lockdown indicators
  - missingness flags
  - city "satellite fingerprints"
- **Validation:** Group K-Fold by city, so no city appears in both training and validation. This mirrors the real test setting of unseen cities.
- **Models:** two CatBoost variants plus LightGBM, blended with weights tuned on out-of-fold predictions
- **Deployment-ready:** final models and all preprocessing artifacts are saved, so new data can be scored without the training set

## Tech stack

Python · pandas · NumPy · scikit-learn · CatBoost · LightGBM · Matplotlib · Seaborn

## Run it

The notebook was built on Kaggle. Attach the competition dataset (`Train.csv`, `Test.csv`, `SampleSubmission.csv`) and update the paths in section 2 if you run it elsewhere.

```bash
pip install pandas numpy scikit-learn catboost lightgbm matplotlib seaborn joblib
jupyter notebook AQI_PM25_Prediction.ipynb
```

## Contents

`AQI_PM25_Prediction.ipynb`: the full pipeline, from EDA through feature engineering, training, blending and the submission, plus a reference list.
