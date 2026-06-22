import sqlite3

with sqlite3.connect('nasa_cmapss_logs.db') as conn:
    cursor = conn.cursor()
    
    cursor.execute("""
                   CREATE TABLE IF NOT EXISTS flight_model_runs (
                       run_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                       timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                       dataset_id TEXT NOT NULL,
                       model_architecture TEXT NOT NULL,
                       learning_rate REAL,
                       max_depth INTEGER, 
                       n_estimators INTEGER,
                       val_score REAL,
                       test_nasa_score REAL
                   );
            """)
    conn.commit()

new_columns = [
    ("subsample", "REAL"),
    ("colsample_bytree", "REAL"),
    ("min_child_weight", "INTEGER"),
    ("reg_alpha", "REAL"),
    ("reg_lambda", "REAL")
]

with sqlite3.connect('nasa_cmapss_logs.db') as conn:
    cursor = conn.cursor()
    
    for col_name, col_type in new_columns:
        try:
            cursor.execute(f"ALTER TABLE flight_model_runs ADD COLUMN {col_name} {col_type}")
            print(f"Column {col_name} added to database")
        except sqlite3.OperationalError:
            print(f"Column {col_name} already exists. Skipping")
    
    conn.commit()

print(f"Database schema verification complete.")