import pandas as pd
import numpy as np 
from xgboost import XGBRegressor
import optuna
import sqlite3
from sklearn.model_selection import GroupKFold
import os

# =====================================================
# ONE-TIME CONVERSION SETUP ( preserved for reference)
# =====================================================

base_columns = ['unit number', 'time, in cycles', 'operational setting 1', 'operational setting 2', 'operational setting 3']
sensor_columns = [f"sensor_{i}" for i in range(1, 22)]
all_columns = base_columns + sensor_columns

outputFolder = "DataParquet"
if not os.path.exists(outputFolder):
    os.makedir(outputFolder)
    print(f"DataParquet, folder has been created")

# df_raw = pd.read_csv("DataTxt/train_FD001.txt", sep=r"\s+", header=None, names=all_columns)
# df_raw.to_parquet(f"{outputFolder}/raw_FD001_train.parquet", index=False)
# print("Raw Data successfully loaded and frozen into a parquet file")
# print(df_raw['sensor_columns'].std())

df_test = pd.read_csv("DataTxt/test_FD001.txt", sep=r"\s+", header=None, names=all_columns)
df_test.to_parquet(f"{outputFolder}/raw_FD001_test.parquet", index=False)
df_rul_answers = pd.read_csv("DataTxt/RUL_FD001.txt", header=None, names=["Remaining Life"])
print("Raw Test Data successfully loaded and frozen into a parquet file.")

def load_and_clean_data(file_path=f"{outputFolder}/raw_FD001_train.parquet") -> pd.DataFrame:
    """
    Loads the frozen Parquet telemetry file and automatically strips away
    the zero-variance dead sensors (1, 5, 10, 16, 18, 19).
    """
    df_loaded = pd.read_parquet(file_path)
    
    # 2. Define the exact dead columns we discovered through variance analysis
    dead_cols = ['sensor_1', 'sensor_5', 'sensor_10', 'sensor_16', 'sensor_18', 'sensor_19']
    
    # 3. Drop them cleanly
    df_clean = df_loaded.drop(columns=dead_cols)
    
    print(f"Pipeline: Successfully loaded {file_path}")
    print(f"Pipeline: Stripped 6 dead sensors. Shape is now {df_clean.shape}")
    
    return df_clean

df = load_and_clean_data()

# Printing the max life cycle for each unit

# max_lifeCycle = df.groupby(['unit number'])['time, in cycles'].max()
# print(max_lifeCycle)

# ===========================
# Target Calculation
# ===========================

def RUL_Calculation(input_df: pd.DataFrame) -> pd.DataFrame:
    
    df_target = input_df.copy()
    
    df_target['max_cycle_temp'] = df_target.groupby(['unit number'])['time, in cycles'].transform('max')
    # Standard Ceiling = 125 cycles 
    
    df_target['True_RUL'] = df_target['max_cycle_temp'] - df_target['time, in cycles']
    df_target['Piecewise_RUL'] = df_target['True_RUL'].clip(upper=125)
    
    final_df = df_target.drop(columns=['True_RUL', 'max_cycle_temp'])
    
    return final_df

df_with_target = RUL_Calculation(df)
    
# print(df_with_target.head())
# print(df_with_target.info())

def correlation(input_df: pd.DataFrame):
    columns_to_check = [col for col in input_df.columns if 'sensor' in col] + ['Piecewise_RUL']
    correlation_matrix = input_df[columns_to_check].corr()
    
    target_correlations = correlation_matrix['Piecewise_RUL'].sort_values()
    
    print(target_correlations)
    
    return

# correlation(df_with_target)

# Negative correlation (best) -- sensor = [11, 4, 15, 17, 2, 3, 8, 13] -- strongest = 11
# Positive correlation (best) -- sensor = [20, 21, 7, 12] -- strongest = 12

def feature_development(input_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = input_df.copy()
    
    filtered_columns = ['sensor_11', 'sensor_4', 'sensor_15', 'sensor_17', 'sensor_2', 'sensor_3', 'sensor_8', 'sensor_13', 'sensor_20',
                        'sensor_21', 'sensor_7', 'sensor_12']
    for column_name in filtered_columns:
        mean_5_col = f"{column_name}_roll_5"
        mean_20_col = f"{column_name}_roll_20"
        divergence = f"{column_name}_divergence"
        trend_acc = f"{column_name}_trend_acc"
        Systemic_Volatility = f"{column_name}_systemic_volatility"
        
        feature_df[mean_5_col] = feature_df.groupby(['unit number'])[column_name].transform(lambda x: x.rolling(window=5).mean())
        feature_df[mean_20_col] = feature_df.groupby(['unit number'])[column_name].transform(lambda x: x.rolling(window=20).mean())
    
        feature_df[divergence] = feature_df[mean_5_col] - feature_df[mean_20_col]
        
        feature_df[trend_acc] = feature_df.groupby(['unit number'])[column_name].diff(periods=5)
        feature_df[Systemic_Volatility] = feature_df.groupby(['unit number'])[column_name].transform(lambda x: x.rolling(window=5).std())
    
    # Fleet-Wide Cross-Sensor Ratio
    feature_df['Cross_Sensor_Ratio'] = feature_df['sensor_11'] / feature_df['sensor_12']
    
    print(f"Features have been engineered successfully.")
    
    # Data Cleaning
    clean_df = feature_df.dropna(subset=['sensor_11_roll_20'])
    clean_df.reset_index(drop=True, inplace=True)
    
    print(f"Data Cleaned Successfully")

    return clean_df

main_df = feature_development(df_with_target)


# print(main_df.head())
# print(main_df.info())

def error_detection(y_true, y_pred):
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    d = y_pred - y_true
    penalties = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    
    total_score = np.sum(penalties)
    avg_score = np.mean(penalties)
    
    print(f"Validation Size: {len(d)} rows")
    print(f"Sum of Early Preds: {np.sum(d<=0)}")
    print(f"Sum of Late Preds: {np.sum(d>0)}")
    
    return total_score

# ======================================
# Making the Model
# ======================================

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

X_train, y_train, X_val, y_val, train_groups = split_data(main_df)

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

train_engine, val_score = train_model(X_train, y_train, X_val, y_val)


# ====================
# Creating Test Set
# ====================

# We already have df_test and df_rul_answers

def test_data_run(trained_model):
    
    df = load_and_clean_data(f"{outputFolder}/raw_FD001_test.parquet")
    df = feature_development(df)
    
    df_last = df.groupby('unit number').last().reset_index()
    X_test = df_last.drop(columns=['unit number', 'time, in cycles'])
    y_test = df_rul_answers['Remaining Life'].values
    
    test_preds = trained_model.predict(X_test)
    
    final_score = error_detection(y_test, test_preds)
    
    return final_score

final_test_score = test_data_run(train_engine)
print(final_test_score)


# ===========================
# OPUTNA
# ===========================

def objective_optuna(trial):
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
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
    }
    
    model_optuna = XGBRegressor (
        **params,
        objective=nasa_objective,
        random_state=42,
        n_jobs=-1
    )
    
    gkf = GroupKFold(n_splits=5)
    group_score = []
    for train_idx, val_idx in gkf.split(X_train, y_train, groups=train_groups):
        
        X_tr_fold = X_train.iloc[train_idx]
        y_tr_fold = y_train.iloc[train_idx]
        
        X_va_fold = X_train.iloc[val_idx]
        y_va_fold = y_train.iloc[val_idx]
        
        model_optuna.fit(X_tr_fold, y_tr_fold)

        preds = model_optuna.predict(X_va_fold)

        score = error_detection(y_va_fold, preds)
        
        group_score.append(score)

    fold_score = np.mean(group_score)
    return fold_score

study = optuna.create_study(direction='minimize')
study.optimize(objective_optuna, n_trials=30)

best_params = study.best_params
best_score = study.best_value

print(f"Running Optuna Model")
optuna_model = XGBRegressor (
    **best_params,
    random_state=42,
    n_jobs=-1,
    objective=nasa_objective
)

optuna_model.fit(X_train, y_train)
final_optuna_test_score = test_data_run(optuna_model)


# =======================
# LOGGING
# =======================
    
def logging_data(model_type: str, params: dict, val_score: float, test_score: float):
    
    lr = params.get('learning_rate', 0.1)
    depth = params.get('max_depth', 5)
    estimators = params.get('n_estimators', 100)
    subsample = params.get('subsample', 1.0)
    colsample = params.get('colsample_bytree', 1.0)
    min_child_weight = params.get('min_child_weight', 1)
    reg_alpha = params.get('reg_alpha', 0.0)
    reg_lambda = params.get('reg_lambda', 1.0)
    
    run_payload = (
        "FD001", model_type, lr, depth, estimators, subsample, colsample, min_child_weight, reg_alpha, reg_lambda, float(val_score), float(test_score)
    )
    
    with sqlite3.connect('nasa_cmapss_logs.db') as conn:
        cursor = conn.cursor()
    
        cursor.execute("""
                        INSERT OR IGNORE INTO flight_model_runs (
                            dataset_id, model_architecture, learning_rate, max_depth, n_estimators, subsample, colsample_bytree, min_child_weight,
                            reg_alpha, reg_lambda, val_score, test_nasa_score
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, run_payload)
        conn.commit()

# logging the data

def log_data_per_test():
    # baseline_params = {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators':100}
    # logging_data("XGBRegressor", baseline_params, val_score, final_test_score)
    
    logging_data("XGBRegressor_Optuna",best_params, best_score, final_optuna_test_score)

log_data_per_test()