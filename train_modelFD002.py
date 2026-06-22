import pandas as pd
import numpy as np
import optuna
import os 
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor, early_stopping
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import RidgeCV
from sklearn.cluster import KMeans

# ===========================================
# Static Constants
# ===========================================

BASE_COLUMNS = ['unit_number', 'time_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3']
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = BASE_COLUMNS + SENSOR_COLUMNS
OUTPUT_FOLDER = "DataParquet"

# ===========================================
# Data Importing
# ===========================================

def load_freeze_data():
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
        print(f"'{OUTPUT_FOLDER}', folder has been created")
        
    df_raw = pd.read_csv("DataTxt/train_FD002.txt", sep=r"\s+", header=None, names=ALL_COLUMNS)
    df_raw.to_parquet(f"{OUTPUT_FOLDER}/raw_FD002_train.parquet", index=False)
    print("Raw Data successfully loaded and frozen into a parquet file")

    df_test = pd.read_csv("DataTxt/test_FD002.txt", sep=r"\s+", header=None, names=ALL_COLUMNS)
    df_test.to_parquet(f"{OUTPUT_FOLDER}/raw_FD002_test.parquet", index=False)
    print("Raw Test Data successfully loaded and frozen into a parquet file.")

# ===========================================
# Loading File
# ===========================================

def load_and_clean_data(file_path=f"{OUTPUT_FOLDER}/raw_FD002_train.parquet") -> pd.DataFrame:
    """
    Loads the frozen Parquet telemetry file and automatically strips away
    the zero-variance dead sensors
    """
    df_loaded = pd.read_parquet(file_path)
    
    # 2. Define the exact dead columns we discovered through variance analysis
    dead_cols = ['sensor_10', 'sensor_15', 'sensor_16']
    
    # 3. Drop them cleanly
    df_clean = df_loaded.drop(columns=dead_cols)
    
    print(f"Pipeline: Successfully loaded {file_path}")
    print(f"Pipeline: Stripped 3 dead sensors. Shape is now {df_clean.shape}")
    
    return df_clean

# ===========================================
# RUL Calculation
# ===========================================
def RUL_Calculation(input_df: pd.DataFrame) -> pd.DataFrame:
    
    df_target = input_df.copy()
    
    df_target['max_cycle_temp'] = df_target.groupby(['unit number'])['time, in cycles'].transform('max')
    # Standard Ceiling = 125 cycles 
    
    df_target['True_RUL'] = df_target['max_cycle_temp'] - df_target['time, in cycles']
    df_target['Piecewise_RUL'] = df_target['True_RUL'].clip(upper=125)
    
    final_df = df_target.drop(columns=['True_RUL', 'max_cycle_temp'])
    
    return final_df

# ===========================================
# Data Processing
# ===========================================

def preprocess_data(input_df: pd.DataFrame) -> pd.DataFrame:
    copy_df = input_df.copy()
    # Making the functions
    copy_df['Theta'] = copy_df['sensor_1'] / 518.67
    copy_df['Delta'] = copy_df['sensor_5'] / 14.696
    
    # Applying
    temperature_cols = ['sensor_2', 'sensor_3', 'sensor_4']
    pressure_cols = ['sensor_7', 'sensor_6', 'sensor_11']
    speed_cols = ['sensor_8', 'sensor_9', 'sensor_13', 'sensor_14'] 
    mass_flow_rate_cols = ['sensor_20', 'sensor_21']
    
    copy_df[temperature_cols] = copy_df[temperature_cols].div(copy_df['Theta'], axis=0)
    copy_df[speed_cols] = copy_df[speed_cols].div(np.sqrt(copy_df['Theta']), axis=0)
    copy_df[pressure_cols] = copy_df[pressure_cols].div(copy_df['Delta'], axis=0)
    
    flow_multiplier = np.sqrt(copy_df['Theta']) / copy_df['Delta']
    copy_df[mass_flow_rate_cols] = copy_df[mass_flow_rate_cols].mul(flow_multiplier, axis=0)
    
    final_df = copy_df.drop(columns=['sensor_1', 'sensor_5', 'Theta', 'Delta'])
    print("Thermodynamic Normalization Complete")
    
    return final_df

def apply_regimes_and_scale(df_train, df_test):
    # 1. Define columns (Same as before)
    sensor_cols = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_6', 'sensor_11', 
                   'sensor_8', 'sensor_9', 'sensor_13', 'sensor_14', 'sensor_20', 'sensor_21']
    ops_cols = ['operational setting 1', 'operational setting 2', 'operational setting 3']

    # 2. Find regimes (Same as before)
    cluster_model = KMeans(n_clusters=6, random_state=42, n_init=10)
    df_train['regime_id'] = cluster_model.fit_predict(df_train[ops_cols])
    df_test['regime_id'] = cluster_model.predict(df_test[ops_cols])
    
    # 3. CRUCIAL FIX: Calculate means and std devs from TRAIN ONLY
    train_means = df_train.groupby('regime_id')[sensor_cols].mean()
    train_stds = df_train.groupby('regime_id')[sensor_cols].std()
    
    train_means_aligned = train_means.loc[df_train['regime_id']].values
    train_stds_aligned = train_stds.loc[df_train['regime_id']].values
    
    test_means_aligned = train_means.loc[df_test['regime_id']].values
    test_stds_aligned = train_stds.loc[df_test['regime_id']].values

    # 4. Perform the standardization ONLY on the sensor columns
    df_train[sensor_cols] = (df_train[sensor_cols] - train_means_aligned) / train_stds_aligned
    df_test[sensor_cols] = (df_test[sensor_cols] - test_means_aligned) / test_stds_aligned
            
    return df_train, df_test

# ===========================================
# Feature Engineering
# ===========================================

def feature_dev(input_df: pd.DataFrame, is_training =True, train_min=None, train_max=None) -> pd.DataFrame:
    feature_df = input_df.copy()
    
    cols_to_keep = ['sensor_11', 'sensor_4', 'sensor_3', 'sensor_2', 'sensor_7' , 'sensor_20', 'sensor_21']
    for colname in cols_to_keep:
        mean_5_col = f"{colname}_roll_5_mean"
        mean_20_col = f"{colname}_roll_20_mean"
        divergence = f"{colname}_divergence"
        trend_acc = f"{colname}_trend_acc"
        systemic_volatility = f"{colname}_systemic_volatility"
        
        # General Feature Engineering
        
        feature_df[mean_5_col] = feature_df.groupby(['unit number'])[colname].transform(lambda x: x.rolling(window=5).mean())
        feature_df[mean_20_col] = feature_df.groupby(['unit number'])[colname].transform(lambda x: x.rolling(window=20).mean())
        feature_df[divergence] = feature_df[mean_20_col] - feature_df[mean_5_col]
        feature_df[trend_acc] = feature_df.groupby(['unit number'])[colname].diff(periods=5)
        feature_df[systemic_volatility] = feature_df.groupby(['unit number'])[colname].transform(lambda x: x.rolling(window=5).std())
    
    feature_df['cross_sensor_ratio'] = feature_df['sensor_11'] / feature_df['sensor_7']
    
    # Health Index
    corr_dict = {
        "sensor_11":0.77,
        "sensor_4":0.73,
        "sensor_3":0.61,
        "sensor_2":0.63,
        "sensor_7":0.50,
        "sensor_20":0.48,
        "sensor_21":0.47
    }
    
    total_corr = sum(corr_dict.values())
    
    # Weight distribution
    weights = {sensor: score / total_corr for sensor, score in corr_dict.items()}
    
    feature_df['Raw_Health'] = 0.0
    for sensor, weight in weights.items():
        if sensor in ['sensor_11', 'sensor_4', 'sensor_3', 'sensor_2']:
            feature_df['Raw_Health'] += -1 * weight * feature_df[sensor]
        
        else:
            feature_df['Raw_Health'] += weight * feature_df[sensor]
            
    if is_training:      
        train_max = feature_df['Raw_Health'].max()
        train_min = feature_df['Raw_Health'].min()
        
        feature_df['Health_Index'] = (feature_df['Raw_Health'] - train_min) / (train_max - train_min)

    else:        
        feature_df['Health_Index'] = (feature_df['Raw_Health'] - train_min) / (train_max - train_min)
        feature_df['Health_Index'] = feature_df['Health_Index'].clip(lower=0.0, upper=1.0)
        
    feature_df = feature_df.drop(columns=['Raw_Health'])
    
    # Cleaning the DataFrame - Take any Roll_20_avg
    feature_df = feature_df.dropna(subset=['sensor_11_roll_20_mean'])
    feature_df = feature_df.reset_index(drop=True)
    
    
    if is_training:
        return feature_df, train_min, train_max
    else:
        return feature_df

# ===========================================
# Error Calc
# ===========================================

def error_detection(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    d = y_pred - y_true
    penalties = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    
    total_score = np.sum(penalties)
    
    print(f"Validation Size: {len(d)} rows")
    print(f"Sum of Early Preds: {np.sum(d<=0)}")
    print(f"Sum of Late Preds: {np.sum(d>0)}")
    
    return total_score

# ===========================================
# SQLITE Logging Feature
# ===========================================

import sqlite3

def log_flight_model_run(dataset_id, architecture, params, val_score, test_score=None):
    """Used for logging experiment results"""
    with sqlite3.connect('nasa_cmapss_logs.db') as conn:
        cursor = conn.cursor()
        
        query = ("""
                       INSERT OR IGNORE INTO flight_model_runs (
                           dataset_id, model_architecture, learning_rate, max_depth, n_estimators, subsample,
                           colsample_bytree, min_child_weight, reg_alpha, reg_lambda,
                           val_score, test_nasa_score
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       """)
        
        values = (
            dataset_id, architecture, 
            params.get('learning_rate'), params.get('max_depth'), params.get('n_estimators'),
            params.get('subsample'), params.get('colsample_bytree'), params.get('min_child_weight'),
            params.get('reg_alpha'), params.get('reg_lambda'),
            val_score, test_score
        )
        
        cursor.execute(query, values)
        conn.commit()

# ===========================================
# Data Splitting
# ===========================================

def split_data(input_df: pd.DataFrame):
    final_df = input_df.copy()
    
    columns_to_drop = ['unit number', 'time, in cycles', 'Piecewise_RUL']
    val_range = list(range(81, 101))
    df_val_set = final_df[final_df['unit number'].isin(val_range)]
    df_train_set = final_df[~final_df['unit number'].isin(val_range)]
    
    train_groups = df_train_set['unit number']
    
    X_train = df_train_set.drop(columns=columns_to_drop)
    y_train = df_train_set['Piecewise_RUL']
    
    X_val = df_val_set.drop(columns=columns_to_drop)
    y_val = df_val_set['Piecewise_RUL']
    
    return X_train, y_train, X_val, y_val, train_groups

# ===========================================
# Objective Creation
# ===========================================

def nasa_objective(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    d = y_pred - y_true
    
    grad = np.zeros_like(d, dtype=float)
    hess = np.zeros_like(d, dtype=float)
    
    early = d < 0 
    grad[early] = -1/13 * np.exp(-d[early] / 13)
    hess[early] = 1/169 * np.exp(-d[early] / 13)
    
    late = d >= 0 
    grad[late] = 1/10 * np.exp(d[late] / 10)
    hess[late] = 1/100 * np.exp(d[late] / 10)
    
    return grad, hess

# ===========================================
# Base Model template
# ===========================================

def train_model(X_train, y_train, X_val, y_val) -> tuple[XGBRegressor, float]:
    # Baseline
    model = XGBRegressor(
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        objective=nasa_objective,
        max_depth=5
    )
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_val)
    score = error_detection(y_val, preds)
    
    print(f"Error Score --> {score}")
    
    return model, score

# ===========================================
# XGB Optuna template
# ===========================================
def create_xgb_objective(X_train ,y_train, train_groups):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),

            'max_depth': trial.suggest_int('max_depth', 3, 5), 
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            
            # --- THE REGULARIZATION CHAINS ---
            # min_child_weight stops the tree from splitting if a leaf has too few samples
            'min_child_weight': trial.suggest_int('min_child_weight', 3, 10),
            # L1/L2 Regularization mathematically penalizes large leaf weights
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        
        gkf = GroupKFold(n_splits=5)
        score = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=train_groups):
            X_tr_fold, X_va_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_fold, y_va_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            opt_model = XGBRegressor(
                **params,
                objective=nasa_objective,
                random_state=42,
                n_jobs=-1,
                early_stopping_rounds=20
            )
            
            opt_model.fit(
                X_tr_fold, y_tr_fold,
                eval_set=[(X_va_fold, y_va_fold)],
                verbose=False
            )
            
            opt_preds = opt_model.predict(X_va_fold)
            opt_score = error_detection(y_va_fold, opt_preds)
            
            score.append(opt_score)
            
        return np.mean(score)
    return objective

# ===========================================
# LGBM Optuna template
# ===========================================
def create_lgbm_objective(X_train,y_train,train_groups):
    def objective_lgbm(trial):
        # LightGBM-specific search space
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 300, step=50),
            'num_leaves': trial.suggest_int('num_leaves', 20, 60), # Crucial for LGBM!
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 0.9),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        
        gkf = GroupKFold(n_splits=5)
        group_score = []
        
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=train_groups):
            
            X_tr_fold = X_train.iloc[train_idx]
            y_tr_fold = y_train.iloc[train_idx]
            X_va_fold = X_train.iloc[val_idx]
            y_va_fold = y_train.iloc[val_idx]
            
            model_lgbm = LGBMRegressor (
                **params,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
            
            model_lgbm.fit(
                X_tr_fold, y_tr_fold,
                eval_set=[(X_va_fold, y_va_fold)],
                callbacks=[early_stopping(stopping_rounds=20, verbose=False)]
            )

            preds = model_lgbm.predict(X_va_fold)
            score = error_detection(y_va_fold, preds)
            group_score.append(score)

        return np.mean(group_score)
    return objective_lgbm

# ===========================================
# Model Stacking
# ===========================================

def model_stack(input_x, input_y, groups):
    xgb_oof = np.zeros(len(input_x))
    lgb_oof = np.zeros(len(input_x))
    
    gkf = GroupKFold(n_splits=5)
    for (train_idx, val_idx) in gkf.split(input_x, input_y, groups=groups):
        X_tr_fold, X_va_fold = input_x.iloc[train_idx], input_x.iloc[val_idx]
        y_tr_fold, y_va_fold = input_y.iloc[train_idx], input_y.iloc[val_idx]
        
        model_xgb = XGBRegressor(
            random_state=42,
            objective=nasa_objective,
            n_jobs=-1,
        ).fit(X_tr_fold, y_tr_fold)
        
        model_lgb = LGBMRegressor(
            random_state=42,
            verbosity=-1
        ).fit(X_tr_fold, y_tr_fold)
    
        xgb_oof[val_idx] = model_xgb.predict(X_va_fold)
        lgb_oof[val_idx] = model_lgb.predict(X_va_fold)
    
    X_meta_train = pd.DataFrame({
        'XGB': xgb_oof,
        'LGB': lgb_oof
    })
    
    ridge_vals = [0.01, 0.1, 1.0, 10.0, 100.0]
    meta_model = RidgeCV(alphas=ridge_vals)
    meta_model.fit(X_meta_train, input_y)
    
    print(f"Optimal Alpha Selected: {meta_model.alpha_}")
    print(f"Ensemble Weights -> XGBoost: {meta_model.coef_[0]:.4f} | LightGBM: {meta_model.coef_[1]:.4f}")
    
    return meta_model

def run_training_pipeline():
    # Loading the data to kick it off.
    load_freeze_data()
    
    # Data Cleaning
    df_train= load_and_clean_data()
    df_test = load_and_clean_data(f"{OUTPUT_FOLDER}/raw_FD002_test.parquet")
    
    # RUL Calculation
    df_train = RUL_Calculation(df_train)
    
    # Pre-processing data
    df_train = preprocess_data(df_train)
    df_test = preprocess_data(df_test)
    
    # Applying regime_id grouping
    df_train, df_test = apply_regimes_and_scale(df_train, df_test)
    
    # Feature engineering
    df_train, global_min, global_max = feature_dev(df_train, True)
    df_test = feature_dev(df_test, False, global_min, global_max)
    
    # Test Set Creation
    X_test = df_test.groupby('unit number').last().reset_index()

    cols_to_drop = ['unit number', 'time, in cycles']
    X_test_clean = X_test.drop(columns=[col for col in cols_to_drop if col in X_test.columns])
    
    df_rul_answers = pd.read_csv("DataTxt/RUL_FD002.txt", header=None, names=["Remaining Life"])
    y_test = df_rul_answers['Remaining Life']
    
    # Data Splitting
    X_train, y_train, X_val, y_val, train_groups = split_data(df_train)
    
    # --------------- Base Model Run --------------- #
    train_engine, val_score = train_model(X_train, y_train, X_val, y_val)
    baseline_params = train_engine.get_params()

    test_preds_base = train_engine.predict(X_test_clean)
    base_test_score = error_detection(y_test, test_preds_base)
    # Run this when required to log base model
    # log_flight_model_run("FD002", "XGBoost", baseline_params, val_score, base_test_score)
    # ---------------        X       --------------- #
    
    # --------------- XGB_Optuna Run --------------- #
    print("Initiating Optuna Study")
    study = optuna.create_study(direction='minimize')
    study.optimize(create_xgb_objective(X_train, y_train, train_groups), n_trials=30)

    opt_xgb_best_model = XGBRegressor(
        **study.best_params,
        random_state=42,
        n_jobs=-1,
        objective=nasa_objective
    )

    opt_xgb_best_model.fit(X_train,y_train)

    test_opt_preds = opt_xgb_best_model.predict(X_test_clean)
    test_opt_score = error_detection(y_test, test_opt_preds)
    xgb_params = study.best_params

    # log_flight_model_run("FD002", "XGBRegressor Optuna", study.best_params, study.best_value, test_opt_score)
    # ---------------        X       --------------- #
    
    # --------------- LGBM_Optuna Run --------------- #
    print("Running LightGBM Optuna Study...")
    study_lgbm = optuna.create_study(direction='minimize')
    study_lgbm.optimize(create_lgbm_objective(X_train, y_train, train_groups), n_trials=30)

    opt_lgbm_best_model = LGBMRegressor(
        **study_lgbm.best_params,
        random_state=42,
        verbosity=-1
    ).fit(X_train, y_train)
    test_lgbm_opt_preds = opt_lgbm_best_model.predict(X_test_clean)
    test_lgbm_opt_score = error_detection(y_test, test_lgbm_opt_preds)

    # Save the params safely!
    lgbm_params = study_lgbm.best_params
    print("LightGBM Params Saved:", lgbm_params)

    #log_flight_model_run("FD002", "LGBMRegressor Optuna", lgbm_params, study_lgbm.best_value, test_lgbm_opt_score)
    # ---------------        X       --------------- #
    
    # Meta Model Creation
    meta_model = model_stack(X_train, y_train, train_groups)
    
    # --------------- Stacked Eval + Testing --------------- #
    full_xgb = XGBRegressor(
        **xgb_params,
        random_state=42,
        n_jobs=-1,
        objective=nasa_objective
    ).fit(X_train, y_train)

    full_lgb = LGBMRegressor(
        **lgbm_params,
        verbosity=-1,
        random_state=42
    ).fit(X_train, y_train)

    val_meta_xgb = full_xgb.predict(X_val)
    val_meta_lgb = full_lgb.predict(X_val)
    X_meta_val = pd.DataFrame({
        'XGB': val_meta_xgb,
        'LGB': val_meta_lgb
    })

    final_stacked_val = meta_model.predict(X_meta_val)
    final_val_stacked_score = error_detection(y_val, final_stacked_val)

    # Stacked model testing
    test_meta_xgb = full_xgb.predict(X_test_clean)
    test_meta_lgb = full_lgb.predict(X_test_clean)
    X_meta_test = pd.DataFrame({
        'XGB': test_meta_xgb,
        'LGB': test_meta_lgb
    })

    stack_params = {
        'ridge_alpha': meta_model.alpha_,
        'xgb_weight': meta_model.coef_[0],
        'lgb_weight': meta_model.coef_[1]
    }

    y_test_stack = meta_model.predict(X_meta_test)
    y_test_stack_score = error_detection(y_test, y_test_stack)

    print(f"Score obtained from evaluation {final_val_stacked_score}")
    print(f"Test Score: {y_test_stack_score}")

    log_flight_model_run("FD002", "model stacking", stack_params, final_val_stacked_score, y_test_stack_score)
    # ---------------        X       --------------- #
    
    # --------------- Saving Model --------------- #
    fd002_artifact = {
    'lvl_0_xgb': full_xgb,
    'lvl_0_lgbm': full_lgb,
    'lvl_1_ridge': meta_model,
    'expected_features': list(X_train.columns)
    }

    outputFolder_model = "ModelFile"
    if not os.path.exists(outputFolder_model):
        os.mkdir(outputFolder_model)
        print(f"ModelFile, folder has been created")

    joblib.dump(fd002_artifact, f'{outputFolder_model}/FD002_champion_ensemble.joblib')
    print(f"FD002 ensemble safely written to disk")
    # ---------------        X       --------------- #
    
    
if __name__ == "__main__":
    
    print("Intializing FD002 Pipeline...")
    run_training_pipeline()
