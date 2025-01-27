import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import build_preprocessor, frequency_encode
import os
import logging
import numpy as np

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("script.log"), logging.StreamHandler()],
)

logging.info("Starting script...")

try:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    logging.info(f"Working directory set to {os.getcwd()}")

    # Load training data
    data = pd.read_csv("usjobs_train.csv")
    logging.info("Training data loaded successfully.")

    # Separate features and target
    X = data.drop(columns=["Mean_Salary"])
    y = data["Mean_Salary"]

    # Identify numerical and categorical columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # One-hot encode the 'Skills' column and add to the dataset
    logging.info("One-hot encoding the 'Skills' column...")
    if "Skills" in X.columns:
        skills_onehot = X["Skills"].str.get_dummies(sep=",")
        logging.info(f"'Skills' one-hot encoded into {skills_onehot.shape[1]} features.")
        X = pd.concat([X, skills_onehot], axis=1).drop(columns=["Skills"])
        categorical_cols.remove("Skills")  # Remove 'Skills' from categorical columns after encoding

    # Build preprocessing pipeline
    preprocessor, categorical_splits = build_preprocessor(
        X, numerical_cols, categorical_cols, one_hot_threshold=55
    )

    # Frequency encode high-cardinality columns
    logging.info("Applying frequency encoding to high-cardinality features...")
    X = frequency_encode(X, categorical_splits["high_cardinality"])

    # Preprocess features
    logging.info("Preprocessing features...")
    X_preprocessed = preprocessor.fit_transform(X)

    # Split the data into training and test sets
    logging.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, test_size=0.2, random_state=42
    )

    # Train a Random Forest model
    logging.info("Training Random Forest model...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Evaluate the model
    logging.info("Evaluating the model on the test set...")
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"Model evaluation metrics:")
    logging.info(f"Mean Absolute Error (MAE): {mae:.2f}")
    logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
    logging.info(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    logging.info(f"R-squared (R2): {r2:.2f}")

    # Perform cross-validation
    logging.info("Performing cross-validation...")
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    logging.info(f"Cross-Validation MAE Scores: {np.abs(cv_scores)}")
    logging.info(f"Average CV MAE: {np.abs(cv_scores).mean():.2f}")

    # Save the model and preprocessor
    logging.info("Saving the Random Forest model and preprocessor...")
    joblib.dump(rf_model, "random_forest_model.pkl")
    joblib.dump(preprocessor, "preprocessor.pkl")
    if "Skills" in data.columns:
        joblib.dump(skills_onehot.columns.tolist(), "skills_onehot_features.pkl")
    logging.info("Model and preprocessor saved successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise

logging.info("Script completed successfully.")
