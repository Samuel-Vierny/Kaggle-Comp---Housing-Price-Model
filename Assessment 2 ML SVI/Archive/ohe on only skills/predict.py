import pandas as pd
import joblib
import os
import logging
from preprocessing import frequency_encode

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("predict.log"), logging.StreamHandler()],
)

logging.info("Starting prediction script...")

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"Working directory set to {os.getcwd()}")

    # Load test data
    test_data = pd.read_csv("usjobs_test.csv")
    logging.info("Test data loaded successfully.")

    # Load preprocessor, model, and features
    preprocessor = joblib.load("preprocessor.pkl")
    model = joblib.load("random_forest_model.pkl")
    skills_onehot_features = joblib.load("skills_onehot_features.pkl")
    logging.info("Preprocessor, Random Forest model, and one-hot features loaded successfully.")

    # Keep the 'ID' column for submission
    ids = test_data["ID"]

    # Drop target column if it exists
    if "Mean_Salary" in test_data.columns:
        test_data = test_data.drop(columns=["Mean_Salary"])

    # One-hot encode the 'Skills' column and align with training features
    logging.info("One-hot encoding the 'Skills' column in test data...")
    if "Skills" in test_data.columns:
        skills_onehot = test_data["Skills"].str.get_dummies(sep=",")
        # Align with the one-hot-encoded features used during training
        for col in skills_onehot_features:
            if col not in skills_onehot.columns:
                skills_onehot[col] = 0  # Add missing columns
        skills_onehot = skills_onehot[skills_onehot_features]
        test_data = pd.concat([test_data, skills_onehot], axis=1).drop(columns=["Skills"])

    # Frequency encode high-cardinality features
    high_cardinality_cols = ["ID", "Job", "Company", "Location", "City", "Sector", "Director", "URL"]
    test_data = frequency_encode(test_data, high_cardinality_cols)

    # Preprocess the test data
    logging.info("Preprocessing test data...")
    X_test_preprocessed = pd.DataFrame(
        preprocessor.transform(test_data),
        columns=[f"feature_{i}" for i in range(preprocessor.transform(test_data).shape[1])],
    )

    # Ensure feature alignment
    logging.info("Aligning features with the training set...")
    training_features = [f"feature_{i}" for i in range(X_test_preprocessed.shape[1])]
    X_test_preprocessed = X_test_preprocessed[training_features]

    # Generate predictions
    logging.info("Generating predictions...")
    test_predictions = model.predict(X_test_preprocessed)

    # Create the submission DataFrame
    submission = pd.DataFrame({"ID": ids, "Mean_Salary": test_predictions})
    submission.to_csv("submission.csv", index=False)
    logging.info("Submission file created successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise

logging.info("Prediction script completed successfully.")
