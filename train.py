import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# dvc:stage: model_training
def train_model(train_data):
    """
    Train a machine learning model on the training data.
    Args:
        train_data: DataFrame with training data
    Returns:
        Trained model object
    """
    logger.info("Training model")
    # Separate features and target
    X_train = train_data.drop('target', axis=1).select_dtypes(include=[np.number])
    y_train = train_data['target']

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # # Log feature importances
    # feature_importance = pd.DataFrame({
    #     'feature': X_train.columns,
    #     'importance': model.feature_importances_
    # }).sort_values('importance', ascending=False)

    # logger.info(f"Top 5 important features: {feature_importance.head(5)}")
    return model