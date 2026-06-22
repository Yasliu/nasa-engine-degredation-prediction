# NASA CMAPSS Predictive Maintenance Pipeline

An end-to-end Machine Learning architecture designed to predict the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset. This repository tracks the progression from baseline tabular optimization (FD001) to advanced thermodynamic feature engineering and multi-model stacking (FD002).

## 🚀 Phase 1: FD001 (The Baseline & Custom Loss Engine)
FD001 operates under a constant sea-level phase, allowing the initial architecture to focus heavily on custom evaluation and temporal feature extraction rather than extreme data normalization.

* **Temporal Feature Engineering:** Engineered rolling averages on highly correlated sensors, alongside divergence, trend acceleration, systemic volatility, and cross-sensor fleet-wide ratios.
* **Custom Asymmetric Objective Function:** Standard MSE/MAE fails to capture the physical reality of engine maintenance: predicting a failure *late* is catastrophically worse than predicting it *early*. A custom objective function was written using exact gradients and Hessians derived from NASA's official scoring penalty.
* **Robust Validation:** Built an Optuna pipeline utilizing 5-fold `GroupKFold` cross-validation (grouped by engine unit) to strictly prevent temporal data leakage during training.
* **Experiment Tracking:** Integrated a custom SQLite logger to track all hyperparameter states, validation scores, and final test results.

## 🌪️ Phase 2: FD002 (Thermodynamics & Meta-Model Stacking)
FD002 introduces extreme volatility by simulating engines flying through 6 distinct operational regimes (varying altitudes and Mach numbers). Standard normalization techniques (like `StandardScaler`) fail entirely here, requiring a physics-based approach.

* **Thermodynamic Normalization:** Extracted base temperature ($\theta$) and pressure ($\delta$) to physically correct fan speed, exhaust gas temperature, and mass flow rates.
* **Regime-Specific Standardization:** Applied K-Means clustering to isolate the 6 flight regimes, scaling the thermodynamic data *locally* within each regime to prevent false normalization across different altitudes.
* **Health Index Generation:** Engineered a consolidated, weighted `Health_Index` feature to provide the model with a smoothed degradation trajectory.
* **Level 1 Meta-Model Stacking:**
    * **Level 0 Base:** XGBoost (capturing global trends) and LightGBM (capturing localized anomalies).
    * **Level 1 Stack:** LightGBM produced superior test scores but exhibited validation volatility. A `RidgeCV` meta-model was implemented to stack the out-of-fold predictions, mathematically balancing XGBoost's stability with LightGBM's precision to significantly drive down the final asymmetric error score.

## ⚙️ Execution
The pipeline is modularized into a production-ready script. It automatically ingests raw text, serializes to Parquet for performance, engineers features, trains the stack, and outputs a deployable `.joblib` artifact.

## Link to Download Data
If you wish to try it out yourself, here is the link of the Nasa CMPASS data.
[NASA_CMPASS_DATA](https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data)

Preferably save the data text files in a folder named `DataTxt`

```bash
python train_modelFD002.py