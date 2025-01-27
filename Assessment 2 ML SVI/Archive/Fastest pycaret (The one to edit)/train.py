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

    # Define best parameters based on previous results
    best_params = {
        'depth': 10,
        'learning_rate': 0.1,
        'iterations': 1000,
        'l2_leaf_reg': 2,
        'bagging_temperature': 0.5,
        'random_seed': 42,
        'verbose': 100
    }

    # Train preliminary model to get feature importance
    logging.info("Training preliminary CatBoost model for feature selection...")
    catboost_model = CatBoostRegressor(**best_params)
    catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    logging.info("Preliminary model training completed.")

    # Extract feature importance
    logging.info("Extracting feature importance...")
    feature_importances = pd.DataFrame({
        'Feature': preprocessor.get_feature_names_out(),
        'Importance': catboost_model.get_feature_importance(train_pool)
    }).sort_values(by='Importance', ascending=False)

    # Log feature importance
    logging.info("Feature importance:\n" + feature_importances.to_string(index=False))

    # Select top features (cumulative importance threshold of 95%)
    cumulative_importance = feature_importances['Importance'].cumsum()
    selected_features = feature_importances.loc[cumulative_importance <= 95, 'Feature']
    logging.info(f"Selected features (95% cumulative importance): {selected_features.tolist()}")

    # Filter training data to selected features
    logging.info("Filtering training data to selected features...")
    X_train_filtered = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())[selected_features]
    X_val_filtered = pd.DataFrame(X_val, columns=preprocessor.get_feature_names_out())[selected_features]

    # Update CatBoost Pool with selected features
    train_pool = Pool(X_train_filtered, y_train)
    val_pool = Pool(X_val_filtered, y_val)

    # Train final model with selected features
    logging.info("Training final CatBoost model...")
    catboost_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
    logging.info("Final model training completed.")

    # Save the model, preprocessor, and selected features
    logging.info("Saving the final model, preprocessor, and selected features...")
    catboost_model.save_model("catboost_final_model.cbm")
    joblib.dump(preprocessor, "preprocessor.pkl")
    joblib.dump(selected_features.tolist(), "selected_features.pkl")
    logging.info("Final model, preprocessor, and selected features saved successfully.")

except Exception as e:
    logging.error(f"An error occurred: {e}", exc_info=True)
    raise

# Log script completion
logging.info("Script completed successfully.")
