import pandas as pd
import joblib
from pycaret.regression import *
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
    data_preprocessed = pd.DataFrame(X_preprocessed, columns=[f'feature_{i}' for i in range(X_preprocessed.shape[1])])
    data_preprocessed['Mean_Salary'] = y.reset_index(drop=True)

    # PyCaret setup
    logging.info("Initializing PyCaret setup...")
    regression_setup = setup(
        data=data_preprocessed,
        target='Mean_Salary',
        train_size=0.8,
        normalize=True,
        session_id=42,
        log_experiment=False
    )

    # Compare models
    logging.info("Comparing models...")
    best_model = compare_models(sort='RMSE')

    # Fine-tune the best model
    logging.info("Tuning the best model...")
    tuned_model = tune_model(best_model, optimize='RMSE')

    # Save the best model
    logging.info("Saving the best PyCaret model and preprocessor...")
    final_model = finalize_model(tuned_model)
    joblib.dump(final_model, "pycaret_best_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    logging.info("Best model and preprocessor saved successfully.")

    # Evaluate the best model
    logging.info("Evaluating the best model...")
    evaluate_model(final_model)

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise

# Log script completion
logging.info("Script completed successfully.")
