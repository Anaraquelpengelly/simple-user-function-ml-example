from data_preprocessing import preprocess_data, clean_data, merge_datasets, extract_features, train_test_split_data
from train import train_model
from evaluate import evaluate_model, generate_predictions, save_model
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Main pipeline function
def run_pipeline(raw_data_path="/Users/anapengelly/Documents/work/etic/etiq2_v0.2.3/simple_class_folder/raw_data.csv",
                 features_path="/Users/anapengelly/Documents/work/etic/etiq2_v0.2.3/simple_class_folder/features.csv",
                 external_data_path="/Users/anapengelly/Documents/work/etic/etiq2_v0.2.3/simple_class_folder/external_data.json",
                 model_output_path="model.pkl",
                 predictions_output_path="predictions.csv",
                 report_output_path="report.md"):
    """
    Run the complete ML pipeline from data preprocessing to model evaluation.
    Args:
        raw_data_path: Path to raw data CSV
        features_path: Path to features CSV
        external_data_path: Path to external data JSON
        model_output_path: Path to save the trained model
        predictions_output_path: Path to save predictions
        report_output_path: Path to save evaluation report
    """
    logger.info("Starting ML pipeline")

    # Data processing steps
    preprocessed_data = preprocess_data(raw_data_path, features_path)
    cleaned_data = clean_data(preprocessed_data)
    merged_data = merge_datasets(cleaned_data, external_data_path)
    features_data = extract_features(merged_data)

    # Split data
    train_data, test_data = train_test_split_data(features_data)

    # Model training and evaluation
    model = train_model(train_data)
    evaluate_model(model, test_data, report_output_path)
    generate_predictions(model, test_data, predictions_output_path)
    save_model(model, model_output_path)

    logger.info("ML pipeline completed successfully")

run_pipeline()