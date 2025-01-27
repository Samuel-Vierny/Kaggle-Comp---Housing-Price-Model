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

logging.info("Starting script...")

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"Working directory set to {os.getcwd()}")

    # Load training data
    data = pd.read_csv("usjobs_train.csv")
    logging.info("Training data loaded successfully.")

    # Separate features and target
    X = data.drop(columns=['Mean_Salary'])
    y = data['Mean_Salary']

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # One-hot encode the 'Skills' column and add to the dataset
    logging.info("One-hot encoding the 'Skills' column...")
    if 'Skills' in X.columns:
        skills_onehot = X['Skills'].str.get_dummies(sep=',')
        logging.info(f"'Skills' one-hot encoded into {skills_onehot.shape[1]} features.")
        X = pd.concat([X, skills_onehot], axis=1).drop(columns=['Skills'])
        categorical_cols.remove('Skills')  # Remove 'Skills' from categorical columns after encoding

    # Build preprocessing pipeline
    preprocessor, categorical_splits = build_preprocessor(X, numerical_cols, categorical_cols, one_hot_threshold=55)

    # Frequency encode high-cardinality columns
    logging.info("Applying frequency encoding to high-cardinality features...")
    X = frequency_encode(X, categorical_splits['high_cardinality'])

    # Preprocess features
    logging.info("Preprocessing features...")
    X_preprocessed = preprocessor.fit_transform(X)

    # Prepare data for PyCaret
    data_preprocessed = pd.DataFrame(X_preprocessed, columns=[f'feature_{i}' for i in range(X_preprocessed.shape[1])])
    data_preprocessed['Mean_Salary'] = y.reset_index(drop=True)

    regression_setup = setup(
        data=data_preprocessed,
        target='Mean_Salary',
        train_size=0.8,
        normalize=True,
        session_id=42,
        fold_strategy='kfold',
        fold=10,
        fold_shuffle=True
    )

    # Compare models
    logging.info("Comparing models and optimizing for MAPE...")
    best_model = compare_models(sort='MAPE')

    # Fine-tune the best model
    logging.info("Tuning the best model...")
    tuned_model = tune_model(best_model, optimize='MAPE', n_iter=10)

    # Finalize and save the best model
    logging.info("Saving the best PyCaret model and preprocessor...")
    final_model = finalize_model(tuned_model)
    joblib.dump(final_model, "pycaret_best_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(skills_onehot.columns.tolist(), "skills_onehot_features.pkl")
    logging.info("Best model and preprocessor saved successfully.")

    # Evaluate the model
    evaluate_model(final_model)

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise

logging.info("Script completed successfully.")
