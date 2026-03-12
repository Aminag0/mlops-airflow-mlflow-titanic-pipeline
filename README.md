README.md
# MLOps Assignment 02 – End-to-End ML Pipeline using Airflow & MLflow

## Overview

This project implements an **end-to-end Machine Learning pipeline** using **Apache Airflow** for workflow orchestration and **MLflow** for experiment tracking and model registry. The pipeline predicts passenger survival using the Titanic dataset.

The system demonstrates how MLOps tools can automate data preprocessing, model training, evaluation, and model lifecycle management.

---

## Technologies Used

- Python
- Apache Airflow
- MLflow
- Scikit-learn
- Pandas
- Joblib

---

## Project Structure


MLOps_Assignment_2
│
├── mlops_airflow_mlflow_pipeline.py
├── Titanic-Dataset.csv
├── requirements.txt
├── README.md
├── Technical_Report.docx
│
└── screenshots
├── dag_list.png
├── dag_view.png
├── val_fail_and_retry.png
├── val_retry.png
├── pipeline_success.png
├── pipeline_graphs_view.png
├── mlflow_run.png
├── mlflow_runs.png
├── model_info_mlflow.png
└── model_registry.png


---

## Pipeline Architecture

The workflow is orchestrated using an **Apache Airflow DAG** that defines the dependencies between different tasks.

### Pipeline Steps

1. **Data Ingestion**
   - Load Titanic dataset
   - Log dataset shape
   - Log missing values
   - Push dataset path using XCom

2. **Data Validation**
   - Check missing percentage in `Age` and `Embarked`
   - Raise exception if missing > 30%
   - Demonstrate retry mechanism

3. **Parallel Processing**
   - Handle missing values
   - Feature engineering (`FamilySize`, `IsAlone`)

4. **Data Encoding**
   - Encode categorical features (`Sex`, `Embarked`)
   - Drop irrelevant columns

5. **Model Training**
   - Train Logistic Regression model
   - Log parameters and dataset details using MLflow

6. **Model Evaluation**
   - Calculate Accuracy, Precision, Recall, and F1-score
   - Log metrics to MLflow

7. **Branching Logic**
   - If accuracy ≥ 0.80 → Register model
   - Else → Reject model

8. **Model Registry**
   - Approved models are registered in **MLflow Model Registry**

---

## Running the Pipeline

### 1. Install Dependencies


pip install -r requirements.txt


---

### 2. Start Airflow


airflow standalone


Airflow UI will be available at:


http://localhost:8080


---

### 3. Start MLflow UI


mlflow ui --backend-store-uri ./mlruns --port 5000


MLflow UI will be available at:


http://localhost:5000


---

### 4. Run the DAG

1. Open Airflow UI
2. Locate DAG:


mlops_airflow_mlflow_pipeline


3. Trigger the DAG manually.

---

## Experiment Tracking

MLflow logs the following information for each run:

### Parameters
- model type
- hyperparameters (`C`, `max_iter`)
- dataset size

### Metrics
- Accuracy
- Precision
- Recall
- F1-score

### Artifacts
- Trained model file

Multiple runs were executed with different hyperparameters to compare model performance.

---

## Experiment Comparison

Three experiments were conducted using different regularization values:

| Run | C Value | Accuracy |
|----|----|----|
| Run 1 | 0.5 | ~0.81 |
| Run 2 | 1.0 | ~0.81 |
| Run 3 | 2.0 | ~0.81 |

All runs produced similar performance results, and the best configuration was selected based on stability.

---

## Model Registry

The best-performing model is automatically registered in **MLflow Model Registry** if its accuracy is greater than or equal to **0.80**.

If accuracy falls below this threshold, the model is rejected and the rejection reason is logged.

---

## Key MLOps Features Demonstrated

- Workflow orchestration using Airflow DAG
- Data validation with retry mechanism
- Parallel task execution
- Experiment tracking using MLflow
- Hyperparameter comparison
- Automated model registration
- Conditional pipeline branching

---

## Conclusion

This project demonstrates how Airflow and MLflow can be integrated to build a reliable and automated machine learning pipeline. The system ensures reproducibility, experiment tracking, and controlled model deployment.

Such pipelines are essential in production machine learning systems where automation, monitoring