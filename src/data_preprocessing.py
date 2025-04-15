import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# DVC annotations for tracking
# dvc:stage: data_preparation
def preprocess_data(raw_data_path, features_path):
    """
    Preprocess the raw data by merging it with feature data.
    Args:
        raw_data_path: Path to the raw data CSV
        features_path: Path to the features CSV
    Returns:
        DataFrame with preprocessed data
    """
    logger.info(f"Preprocessing data from {raw_data_path} and {features_path}")
    raw_data = pd.read_csv(raw_data_path)
    features = pd.read_csv(features_path)

    # Merge data based on ID column
    preprocessed = pd.merge(raw_data, features, on='id', how='inner')

    # Handle missing values
    preprocessed.fillna(preprocessed.mean(numeric_only=True), inplace=True)
    logger.info(f"Preprocessed data shape: {preprocessed.shape}")
    return preprocessed


# dvc:stage: clean_data
def clean_data(preprocessed_data):
    """
    Clean the preprocessed data by removing outliers and standardizing formats.
    Args:
        preprocessed_data: DataFrame with preprocessed data
    Returns:
        DataFrame with cleaned data
    """
    logger.info("Cleaning preprocessed data")
    # Create a copy to avoid modifying the original
    cleaned = preprocessed_data.copy()

    # Remove outliers (simple z-score method)
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    z_scores = np.abs((cleaned[numeric_cols] - cleaned[numeric_cols].mean()) / cleaned[numeric_cols].std())
    # cleaned = cleaned[(z_scores < 3).all(axis=1)]

    # Convert categorical columns to proper format
    cat_cols = cleaned.select_dtypes(include=['object']).columns
    for col in cat_cols:
        cleaned[col] = cleaned[col].str.lower().str.strip()

    logger.info(f"Cleaned data shape: {cleaned.shape}")
    return cleaned


# dvc:stage: merge_datasets
def merge_datasets(cleaned_data, external_data_path):
    """
    Merge the cleaned data with external data.
    Args:
        cleaned_data: DataFrame with cleaned data
        external_data_path: Path to external data JSON
    Returns:
        DataFrame with merged data
    """
    logger.info(f"Merging cleaned data with external data from {external_data_path}")
    # Load the external data
    external_df = pd.read_json(external_data_path, lines=True)

    # Merge with cleaned data
    merged = pd.merge(cleaned_data, external_df, on='id', how='left', suffixes=('', '_ext'))
    logger.info(f"Merged data shape: {merged.shape}")
    return merged


# dvc:stage: feature_extraction
def extract_features(merged_data):
    """
    Extract and engineer features from the merged data.
    Args:
        merged_data: DataFrame with merged data
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting features from merged data")
    # Create a copy to avoid modifying the original
    features = merged_data.copy()

    # Feature engineering examples
    # 1. Create interaction features
    numeric_cols = features.select_dtypes(include=[np.number]).columns[:5]  # Limit to first 5 for example
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i + 1:]:
            features[f"{col1}_{col2}_interaction"] = features[col1] * features[col2]

    # 2. Create categorical encodings
    cat_cols = features.select_dtypes(include=['object']).columns
    for col in cat_cols:
        features[f"{col}_encoded"] = features[col].astype('category').cat.codes

    # 3. Create aggregate features
    if 'date' in features.columns:
        features['date'] = pd.to_datetime(features['date'])
        features['year'] = features['date'].dt.year
        features['month'] = features['date'].dt.month
        features['day_of_week'] = features['date'].dt.dayofweek

    logger.info(f"Features data shape: {features.shape}")
    return features


# dvc:stage: data_split
def train_test_split_data(features_data, test_size=0.2, random_state=42):
    """
    Split the features data into training and testing sets.
    Args:
        features_data: DataFrame with feature data
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
    Returns:
        train_data, test_data DataFrames
    """
    logger.info(f"Splitting data with test_size={test_size}")
    # Make sure we have a target column (using 'target' as placeholder)
    if 'target' not in features_data.columns:
        # For this example, create a dummy target if it doesn't exist
        logger.warning("Target column not found, creating a dummy target")
        features_data['target'] = np.random.randint(0, 2, size=len(features_data))

    # Separate features and target
    X = features_data.drop('target', axis=1)
    y = features_data['target']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Reassemble DataFrames
    train_data = X_train.copy()
    train_data['target'] = y_train
    test_data = X_test.copy()
    test_data['target'] = y_test

    logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    return train_data, test_data