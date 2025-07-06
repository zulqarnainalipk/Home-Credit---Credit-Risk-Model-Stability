# Home Credit Risk Model Pipeline

This repository contains a modularized and refactored version of the Home Credit Risk Model Pipeline, originally developed as a Jupyter notebook. The pipeline focuses on data preprocessing, feature engineering, and ensemble modeling to predict credit risk.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project provides a comprehensive approach to building and assessing credit risk models. It leverages various machine learning algorithms, including LightGBM and CatBoost, combined with ensemble techniques for robust predictions. The pipeline emphasizes data integrity, feature relevance, and model stability, which are crucial elements in credit risk assessment.

## Features

- **Data Preprocessing**: Includes cleaning data, handling missing values, and optimizing memory usage for efficient computation.
- **Feature Engineering**: Extracts meaningful insights from raw data using advanced techniques to enhance model predictive power.
- **Model Training**: Supports training multiple machine learning models like LightGBM and CatBoost to capture complex relationships and patterns.
- **Ensemble Learning**: Combines predictions from various models using a custom Voting Model to achieve higher accuracy and stability.
- **Modular Design**: Code is organized into distinct modules for better maintainability and readability.

## Project Structure

```
home-credit-risk-model/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Contains data cleaning, type conversion, and filtering logic
│   ├── feature_aggregation.py   # Handles feature aggregation from various data sources
│   ├── data_loader.py           # Functions for reading single and multiple parquet files
│   ├── feature_engineering.py   # Combines and engineers features from different data depths
│   ├── utils.py                 # Utility functions like memory reduction and pandas conversion
│   └── ensemble_model.py        # Custom VotingModel for ensemble predictions
├── data/                        # Placeholder for raw and processed data (e.g., parquet files)
├── models/                      # Placeholder for trained machine learning models
├── notebooks/                   # Original Jupyter notebooks and exploratory analysis
├── .gitignore                   # Specifies intentionally untracked files to ignore
├── requirements.txt             # Lists project dependencies
└── README.md                    # Project overview and documentation
```

## Setup and Installation

To set up the project, follow these steps:

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/home-credit-risk-model.git
    cd home-credit-risk-model
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

This project is designed to be run in two main modes: `train` and `predict`. The `main_pipeline.py` script orchestrates these processes.

### Training Mode

In training mode, the pipeline reads raw data, performs preprocessing and feature engineering, and prepares the data for model training. Model training itself is currently a placeholder and would typically involve a separate script or module that utilizes the processed data.

To run the pipeline in training mode (data processing only):

```bash
python -c "from src.main_pipeline import run_pipeline; run_pipeline(mode=\'train\')"
```

*Note: The actual model training logic (e.g., `lgb.LGBMClassifier().fit(df_train, y)`) from the original notebook is commented out in `main_pipeline.py` and should be implemented in a dedicated training script within the `models/` directory or a separate training module.* 

### Prediction Mode

In prediction mode, the pipeline loads pre-trained models, processes new test data, and generates predictions. It also applies a post-processing step to adjust scores based on a defined condition.

To run the pipeline in prediction mode:

```bash
python -c "from src.main_pipeline import run_pipeline; run_pipeline(mode=\'predict\')"
```

*Note: This mode requires the pre-trained models and `notebook_info.joblib` files to be available in the `/kaggle/input/homecredit-models-public/` directory, as referenced in the original notebook. For local execution, you would need to download these files and adjust the `ROOT` path in `main_pipeline.py` accordingly.*

## Data

The project expects data in Parquet format, organized in `parquet_files/train` and `parquet_files/test` directories within the `ROOT` path. The original notebook refers to data from a Kaggle competition. You will need to ensure your data is structured similarly or modify the `ROOT`, `TRAIN_DIR`, and `TEST_DIR` variables in `src/main_pipeline.py` to point to your data location.

## Models

Pre-trained LightGBM and CatBoost models are expected to be loaded from `joblib` files. The paths to these models are currently hardcoded to Kaggle input paths. For local development, you would need to download these models and update the paths in `src/main_pipeline.py`.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is open-source and available under the MIT License. See the `LICENSE` file for more details. (Note: A `LICENSE` file is not included in this refactoring, but it's good practice to add one.)


