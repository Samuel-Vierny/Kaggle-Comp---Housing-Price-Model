import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor, Pool
from preprocessing import build_preprocessor, frequency_encode
import os
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("script.log"), logging.StreamHandler()]
)

# Log script start
logging.info("Starting script...")

try:
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"Working directory set to {os.getcwd()}")

    # Load data
    logging.info("Loading training data...")
    data = pd.read_csv("usjobs_train.csv")
    logging.info("Data loaded successfully.")

    # Separate features and target
    logging.info("Splitting features and target...")
    X = data.drop(columns=['Mean_Salary'])
    y = data['Mean_Salary']

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Build preprocessing pipeline
    logging.info("Building preprocessing pipeline...")
    preprocessor, categorical_splits = build_preprocessor(data, numerical_cols, categorical_cols)

    # Frequency encode high-cardinality columns
    logging.info("Applying frequency encoding to high-cardinality features...")
    X = frequency_encode(X, categorical_splits['high_cardinality'])

    # Preprocess features
    logging.info("Preprocessing features...")
    X_preprocessed = preprocessor.fit_transform(X)
    logging.info("Feature preprocessing completed.")

    # Train-test split
    logging.info("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

    # Convert data to CatBoost Pool
    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    # Define parameter grid for grid search
    param_grid = {
        'depth': [8, 10, 12],                  # Depth of the trees
        'learning_rate': [0.9, 0.1, 0.13], # Step size for weight updates
        'iterations': [1000],          # Number of boosting iterations
        'l2_leaf_reg': [1, 2, 3],           # Regularization strength
        'bagging_temperature': [0, 0.5, 1]    # Overfitting control
    }

    # Perform grid search
    logging.info("Starting grid search for hyperparameter tuning...")
    catboost_model = CatBoostRegressor(verbose=0, random_state=42)
    grid_search_result = catboost_model.grid_search(
        param_grid,
        train_pool,
        cv=3,  # Cross-validation folds
        partition_random_seed=42,
        refit=True
    )
    logging.info(f"Grid search completed. Best parameters: {grid_search_result['params']}")
    logging.info(f"Best validation RMSE: {grid_search_result['cv_results']['test-RMSE-mean'][-1]}")

    # Save the best model and preprocessor
    logging.info("Saving the best model and preprocessor...")
    catboost_model.save_model("catboost_best_model.cbm")
    joblib.dump(preprocessor, "preprocessor.pkl")
    logging.info("Best model and preprocessor saved successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise

# Log script completion
logging.info("Script completed successfully.")
