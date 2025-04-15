import pandas as pd
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# dvc:stage: model_evaluation
def evaluate_model(model, test_data, output_path="report.md"):
    """
    Evaluate the trained model on the test data and generate a report.
    Args:
        model: Trained model object
        test_data: DataFrame with test data
        output_path: Path to save the evaluation report
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model")
    # Separate features and target
    X_test = test_data.drop('target', axis=1).select_dtypes(include=[np.number])
    y_test = test_data['target']

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    # Generate report
    report_content = f"""# Model Evaluation Report
## Metrics
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1']:.4f}

## Model Details
- Type: Random Forest
- Parameters: n_estimators=100

## Data Information
- Test data size: {len(test_data)} samples
- Features: {X_test.shape[1]} numeric features
"""

    # Save report to file
    with open(output_path, 'w') as f:
        f.write(report_content)

    logger.info(f"Evaluation report saved to {output_path}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics


# dvc:stage: predictions
def generate_predictions(model, test_data, output_path="predictions.csv"):
    """
    Generate predictions using the trained model and save them to a CSV.
    Args:
        model: Trained model object
        test_data: DataFrame with test data
        output_path: Path to save the predictions
    Returns:
        DataFrame with predictions
    """
    logger.info("Generating predictions")
    # Separate features
    X_test = test_data.drop('target', axis=1).select_dtypes(include=[np.number])

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    # Create DataFrame with predictions
    pred_df = pd.DataFrame({
        'id': test_data.get('id', pd.Series(range(len(test_data)))),
        'prediction': predictions
    })

    # Add probability columns for each class
    for i, col in enumerate(model.classes_):
        pred_df[f'probability_class_{col}'] = probabilities[:, i]

    # Save to CSV
    pred_df.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")
    return pred_df


# dvc:stage: model_save
def save_model(model, output_path="model.pkl"):
    """
    Save the trained model to disk.
    Args:
        model: Trained model object
        output_path: Path to save the model
    Returns:
        Path to the saved model
    """
    logger.info(f"Saving model to {output_path}")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # Save the model
    with open(output_path, 'wb') as f:
        pickle.dump(model, f)

    # Get file size
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
    logger.info(f"Model saved ({file_size:.2f} MB)")
    return output_path
