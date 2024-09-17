# AutoML Pipeline for Model Selection and Versioning

## Overview

This project implements an automated machine learning (AutoML) pipeline to select the best model, tune its hyperparameters, and save it with detailed information, including model name, parameters, evaluation metrics, and version number. The pipeline uses Python and scikit-learn and is designed for simplicity and extendability.

The pipeline currently supports:
- Random Forest
- Support Vector Machine (SVM)
- Logistic Regression

The best model is saved as a versioned object, making it easy to track changes over time.

## Project Structure

```
.
├── automl_pipeline.py   # Main code file for the AutoML pipeline
├── README.md            # Project description and usage
└── models               # Folder where versioned model files will be saved
```

## How It Works

1. **Dataset**: The pipeline uses the Iris dataset (loaded from `sklearn`).
2. **Model Selection**: The pipeline trains three models (`RandomForest`, `SVM`, and `LogisticRegression`), tunes their hyperparameters using grid search, and selects the best model based on test accuracy.
3. **Model Saving**: The best model is saved as an object, which contains:
   - Model Name
   - Initial and Best Hyperparameters
   - Evaluation Metrics (Accuracy and Classification Report)
   - Version Number
4. **Version Control**: Each saved model is assigned a version number, allowing you to keep track of different model versions easily.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Drakunal/AutoML
   cd AutoML
   ```

2. Install dependencies:
   <!--```bash
   pip install -r requirements.txt
   ```-->

   The key dependencies are:
   - `scikit-learn`
   - `joblib`

## Running the Pipeline

Run the pipeline using the following command:

```bash
python automl_pipeline.py
```

The pipeline will:
- Load and preprocess the Iris dataset.
- Train, tune, and evaluate three different models.
- Select the best-performing model and save it as a versioned object in the `models/` directory.

## Saved Models

Each best model will be saved in the `models/` directory with a filename format like:
```
RandomForest_v1.pkl
SVM_v2.pkl
```

These files contain all the information about the model, including:
- Model Name
- Parameters and Best Parameters
- Evaluation Metrics
- Version Number

## Loading and Inspecting Saved Models

To load a saved model and inspect its contents, use the following code:

```python
import joblib

# Load the saved model
model_file = 'models/RandomForest_v1.pkl'  # Replace with the actual file path
loaded_model = joblib.load(model_file)

# Access the model details
print(f"Model Name: {loaded_model.model_name}")
print(f"Version: {loaded_model.version}")
print(f"Parameters: {loaded_model.params}")
print(f"Best Parameters: {loaded_model.best_params}")
print(f"Evaluation Metrics: {loaded_model.evaluation_metrics}")

# Access the actual trained model
trained_model = loaded_model.model
print(trained_model)
```

## Key Features

- **Automated Model Selection**: Automatically trains and tunes multiple models to select the best one.
- **Versioned Model Saving**: Each model is saved with a version number for easy tracking.
- **Hyperparameter Tuning**: Uses grid search for hyperparameter tuning to find the best configuration for each model.
- **Model Evaluation**: Reports accuracy and detailed classification metrics for the best model.

## Future Work

- Add support for additional machine learning algorithms.
- Automate feature selection and lightweight feature engineering.
- Integrate with other datasets for a more general pipeline.

## License

This project is licensed under the MIT License. Feel free to use and modify the code.
```

