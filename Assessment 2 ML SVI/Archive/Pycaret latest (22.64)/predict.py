import pandas as pd
import joblib
import os
import logging
from catboost import CatBoostRegressor
from preprocessing import frequency_encode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("predict.log"),
        logging.StreamHandler()
    ]
)

# Log script start
logging.info("Starting prediction script...")

try:
    # Change to script directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"Working directory set to {os.getcwd()}")

    # Load test data
    logging.info("Loading test data...")
    test_data = pd.read_csv("usjobs_test.csv")
    logging.info("Test data loaded successfully.")

    # Load preprocessor, model, and selected features
    logging.info("Loading saved preprocessor, model, and selected features...")
    preprocessor = joblib.load("preprocessor.pkl")
    selected_features = joblib.load("selected_features.pkl")
    catboost_model = CatBoostRegressor()
    catboost_model.load_model("catboost_final_model.cbm")  # Use CatBoost's load_model
    logging.info("Preprocessor, model, and selected features loaded successfully.")

    # Keep the 'ID' column for submission
    ids = test_data['ID']

    # Drop target column if it exists
    if 'Mean_Salary' in test_data.columns:
        test_data = test_data.drop(columns=['Mean_Salary'])

    # Identify high-cardinality columns (ensure they are properly encoded)
    high_cardinality_cols = ['ID', 'Job', 'Company', 'Location', 'City', 'Skills', 'Sector', 'Director', 'URL']
    logging.info("Applying frequency encoding to high-cardinality features...")
    test_data = frequency_encode(test_data, high_cardinality_cols)

    # Log data types after frequency encoding
    logging.info("Data types after frequency encoding:")
    logging.info(test_data.dtypes)

    # Preprocess the test data
    logging.info("Preprocessing test data...")
    X_test_preprocessed = preprocessor.transform(test_data)

    # Ensure dense output if sparse matrix is returned
    if hasattr(X_test_preprocessed, "toarray"):
        X_test_preprocessed = X_test_preprocessed.toarray()

    # Convert preprocessed data to DataFrame and align with selected features
    logging.info("Converting preprocessed data to DataFrame...")
    X_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=preprocessor.get_feature_names_out())
    logging.info("Aligning test data with selected features...")
    X_test_preprocessed = X_test_preprocessed[selected_features]

    # Ensure all columns are numeric
    logging.info("Ensuring all columns are numeric...")
    if not X_test_preprocessed.applymap(lambda x: isinstance(x, (int, float))).all().all():
        # Identify problematic columns
        non_numeric_cols = X_test_preprocessed.select_dtypes(exclude=[int, float]).columns
        logging.error(f"Non-numeric columns detected after preprocessing: {non_numeric_cols}")
        raise ValueError("Non-numeric data detected in preprocessed test data!")

    # Generate predictions
    logging.info("Generating predictions...")
    test_predictions = catboost_model.predict(X_test_preprocessed)
    logging.info("Predictions generated successfully.")

    # Create the submission DataFrame
    logging.info("Creating submission file...")
    submission = pd.DataFrame({
        'ID': ids,
        'Mean_Salary': test_predictions
    })

    # Save the submission file
    submission.to_csv("submission.csv", index=False)
    logging.info("Submission file 'submission.csv' has been created successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise

# Log script completion
logging.info("Prediction script completed successfully.")
