import pandas as pd
import logging
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("preprocessing.log"),  # Log to a file
        logging.StreamHandler()  # Log to the terminal
    ]
)

def build_preprocessor(data, numerical_cols, categorical_cols, one_hot_threshold=139):
    """
    Build a preprocessing pipeline for both training and test datasets.
    
    Parameters:
    - data (pd.DataFrame): Dataset to preprocess.
    - numerical_cols (list): List of numerical column names.
    - categorical_cols (list): List of categorical column names.
    - one_hot_threshold (int): Threshold for switching from one-hot to label encoding.
    
    Returns:
    - preprocessor: A preprocessing pipeline.
    - categorical_splits: Dictionary containing low and high cardinality features.
    """
    logging.info("Building preprocessing pipeline...")
    
    # Handle missing values for numerical columns
    logging.info("Setting up numerical transformer...")
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
        ('scaler', StandardScaler())                   # Scale numerical values
    ])
    logging.info("Numerical transformer configured.")

    # Compute unique counts using the data DataFrame
    logging.info("Computing unique value counts for categorical features...")
    unique_counts = {col: data[col].nunique() for col in categorical_cols}
    logging.info(f"Unique counts: {unique_counts}")

    # Split categorical features by cardinality
    logging.info(f"Splitting features by cardinality with threshold = {one_hot_threshold}...")
    low_cardinality_cols = [col for col, count in unique_counts.items() if count <= one_hot_threshold]
    high_cardinality_cols = [col for col, count in unique_counts.items() if count > one_hot_threshold]
    logging.info(f"Low-cardinality features: {low_cardinality_cols}")
    logging.info(f"High-cardinality features: {high_cardinality_cols}")

    # One-hot encoding for low-cardinality features
    logging.info("Setting up categorical transformer for low-cardinality features...")
    categorical_one_hot = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),  # Handle missing values
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # One-hot encoding (dense output)
    ])
    logging.info("Categorical transformer configured.")

    # Combine numerical and low-cardinality pipelines
    logging.info("Combining numerical and categorical transformers into a single preprocessor...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat_low', categorical_one_hot, low_cardinality_cols)
        ],
        remainder='passthrough'
    )
    logging.info("Preprocessor built successfully.")
    
    return preprocessor, {'low_cardinality': low_cardinality_cols, 'high_cardinality': high_cardinality_cols}

def frequency_encode(data, high_cardinality_cols):
    """
    Apply frequency encoding to high-cardinality features.
    
    Parameters:
    - data (pd.DataFrame): Dataset to encode.
    - high_cardinality_cols (list): High cardinality features to encode.
    
    Returns:
    - pd.DataFrame: Encoded dataset.
    """
    logging.info("Applying frequency encoding to high-cardinality features...")
    for col in high_cardinality_cols:
        logging.info(f"Encoding column: {col}")
        freq_map = data[col].value_counts(normalize=True).to_dict()
        data[col] = data[col].map(freq_map).fillna(0)
    logging.info("Frequency encoding completed.")
    return data
