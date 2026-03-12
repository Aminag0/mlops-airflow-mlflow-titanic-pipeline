
from datetime import datetime, timedelta
import os
import json
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from airflow import DAG
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator


BASE_DIR = os.path.expanduser("~/mlops_assignment_2")
DATA_PATH = os.path.join(BASE_DIR, "data", "Titanic-Dataset.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
mlflow.set_experiment("Titanic_Survival_Experiments")


def ingest_data(**context):
    df = pd.read_csv(DATA_PATH)

    print("\n=== DATA INGESTION ===")
    print(f"Dataset path: {DATA_PATH}")
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values count:")
    print(df.isnull().sum())

    context["ti"].xcom_push(key="dataset_path", value=DATA_PATH)
    context["ti"].xcom_push(key="dataset_rows", value=int(df.shape[0]))
    context["ti"].xcom_push(key="dataset_cols", value=int(df.shape[1]))


def validate_data(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(task_ids="data_ingestion", key="dataset_path")

    df = pd.read_csv(dataset_path)

    age_missing_pct = df["Age"].isnull().mean() * 100
    embarked_missing_pct = df["Embarked"].isnull().mean() * 100

    print("\n=== DATA VALIDATION ===")
    print(f"Age missing percentage: {age_missing_pct:.2f}%")
    print(f"Embarked missing percentage: {embarked_missing_pct:.2f}%")

    retry_flag_path = os.path.join(PROCESSED_DIR, "validation_retry_flag.txt")

    if not os.path.exists(retry_flag_path):
        with open(retry_flag_path, "w") as f:
            f.write("failed_once")
        raise ValueError("Intentional failure for retry demonstration in data_validation task.")

    if age_missing_pct > 30:
        raise ValueError(f"Age missing percentage is too high: {age_missing_pct:.2f}%")

    if embarked_missing_pct > 30:
        raise ValueError(f"Embarked missing percentage is too high: {embarked_missing_pct:.2f}%")

    print("Validation passed successfully.")


def handle_missing_values(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(task_ids="data_ingestion", key="dataset_path")

    df = pd.read_csv(dataset_path)

    print("\n=== HANDLE MISSING VALUES ===")
    print("Before filling missing values:")
    print(df[["Age", "Embarked"]].isnull().sum())

    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    print("\nAfter filling missing values:")
    print(df[["Age", "Embarked"]].isnull().sum())

    missing_output_path = os.path.join(PROCESSED_DIR, "missing_values_handled.csv")
    df.to_csv(missing_output_path, index=False)

    ti.xcom_push(key="missing_output_path", value=missing_output_path)
    print(f"Saved file: {missing_output_path}")


def feature_engineering(**context):
    ti = context["ti"]
    dataset_path = ti.xcom_pull(task_ids="data_ingestion", key="dataset_path")

    df = pd.read_csv(dataset_path)

    print("\n=== FEATURE ENGINEERING ===")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    feature_output_path = os.path.join(PROCESSED_DIR, "feature_engineered.csv")
    df.to_csv(feature_output_path, index=False)

    ti.xcom_push(key="feature_output_path", value=feature_output_path)
    print("Created columns: FamilySize, IsAlone")
    print(f"Saved file: {feature_output_path}")


def encode_data(**context):
    ti = context["ti"]

    missing_path = ti.xcom_pull(task_ids="handle_missing_values", key="missing_output_path")
    feature_path = ti.xcom_pull(task_ids="feature_engineering", key="feature_output_path")

    df_missing = pd.read_csv(missing_path)
    df_feature = pd.read_csv(feature_path)

    df_missing["FamilySize"] = df_feature["FamilySize"]
    df_missing["IsAlone"] = df_feature["IsAlone"]

    print("\n=== DATA ENCODING ===")
    print("Encoding categorical columns: Sex, Embarked")

    df_missing["Sex"] = df_missing["Sex"].map({"male": 0, "female": 1})
    df_missing["Embarked"] = df_missing["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    columns_to_drop = ["PassengerId", "Name", "Ticket", "Cabin"]
    existing_drop_cols = [col for col in columns_to_drop if col in df_missing.columns]
    df_missing = df_missing.drop(columns=existing_drop_cols)

    encoded_output_path = os.path.join(PROCESSED_DIR, "encoded_data.csv")
    df_missing.to_csv(encoded_output_path, index=False)

    ti.xcom_push(key="encoded_output_path", value=encoded_output_path)
    print(f"Dropped columns: {existing_drop_cols}")
    print(f"Saved encoded dataset: {encoded_output_path}")


def train_model(**context):
    ti = context["ti"]
    encoded_path = ti.xcom_pull(task_ids="data_encoding", key="encoded_output_path")

    df = pd.read_csv(encoded_path)

    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = context["params"]
    C_value = params.get("C", 1.0)
    max_iter = params.get("max_iter", 200)

    print("\n=== MODEL TRAINING ===")
    print(f"Using Logistic Regression with C={C_value}, max_iter={max_iter}")

    with mlflow.start_run(run_name=f"logreg_C_{C_value}") as run:
        model = LogisticRegression(C=C_value, max_iter=max_iter, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_value)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("dataset_rows", int(df.shape[0]))
        mlflow.log_param("dataset_cols", int(df.shape[1]))

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        model_path = os.path.join(MODEL_DIR, f"logreg_model_C_{C_value}.joblib")
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(model, name="titanic_logistic_model")

        test_data_path = os.path.join(PROCESSED_DIR, "test_data.csv")
        X_test_out = X_test.copy()
        X_test_out["Survived"] = y_test.values
        X_test_out.to_csv(test_data_path, index=False)

        run_info = {
            "run_id": run.info.run_id,
            "model_path": model_path,
            "test_data_path": test_data_path,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        run_info_path = os.path.join(PROCESSED_DIR, "run_info.json")
        with open(run_info_path, "w") as f:
            json.dump(run_info, f, indent=4)

        ti.xcom_push(key="run_info_path", value=run_info_path)
        ti.xcom_push(key="accuracy", value=float(accuracy))

        print(f"Model saved: {model_path}")
        print(f"Run info saved: {run_info_path}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")


def evaluate_model(**context):
    ti = context["ti"]
    run_info_path = ti.xcom_pull(task_ids="model_training", key="run_info_path")

    with open(run_info_path, "r") as f:
        run_info = json.load(f)

    print("\n=== MODEL EVALUATION ===")
    print(f"Run ID: {run_info['run_id']}")
    print(f"Accuracy: {run_info['accuracy']:.4f}")
    print(f"Precision: {run_info['precision']:.4f}")
    print(f"Recall: {run_info['recall']:.4f}")
    print(f"F1-score: {run_info['f1_score']:.4f}")

    ti.xcom_push(key="accuracy", value=float(run_info["accuracy"]))


def choose_branch(**context):
    ti = context["ti"]
    accuracy = ti.xcom_pull(task_ids="model_evaluation", key="accuracy")

    print("\n=== BRANCHING DECISION ===")
    print(f"Accuracy received: {accuracy:.4f}")

    if accuracy >= 0.80:
        print("Decision: register_model")
        return "register_model"
    else:
        print("Decision: reject_model")
        return "reject_model"


def register_model_task(**context):
    ti = context["ti"]
    run_info_path = ti.xcom_pull(task_ids="model_training", key="run_info_path")

    with open(run_info_path, "r") as f:
        run_info = json.load(f)

    model_uri = f"runs:/{run_info['run_id']}/titanic_logistic_model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name="TitanicSurvivalModel"
    )

    registry_log_path = os.path.join(PROCESSED_DIR, "model_registry_log.txt")
    with open(registry_log_path, "a") as f:
        f.write(
            f"REGISTERED | run_id={run_info['run_id']} | "
            f"accuracy={run_info['accuracy']:.4f} | "
            f"model_name=TitanicSurvivalModel | "
            f"model_version={registered_model.version}\n"
        )

    print("\n=== MODEL REGISTRATION ===")
    print("Model registered in MLflow Model Registry.")
    print(f"Registered model name: TitanicSurvivalModel")
    print(f"Registered model version: {registered_model.version}")
    print(f"Registry log file: {registry_log_path}")

def reject_model_task(**context):
    ti = context["ti"]
    accuracy = ti.xcom_pull(task_ids="model_evaluation", key="accuracy")

    rejection_log_path = os.path.join(PROCESSED_DIR, "model_rejection_log.txt")
    with open(rejection_log_path, "a") as f:
        f.write(
            f"REJECTED | accuracy={accuracy:.4f} | reason=Accuracy below threshold 0.80\n"
        )

    print("\n=== MODEL REJECTION ===")
    print(f"Model rejected because accuracy {accuracy:.4f} is below 0.80")
    print(f"Rejection log file: {rejection_log_path}")


default_args = {
    "owner": "amina",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="mlops_airflow_mlflow_pipeline",
    default_args=default_args,
    description="Titanic end-to-end ML pipeline with Airflow and MLflow",
    start_date=datetime(2026, 3, 12),
    schedule=None,
    catchup=False,
    params={"C": 1.0, "max_iter": 200},
    tags=["assignment", "mlops", "titanic"],
) as dag:

    start = EmptyOperator(task_id="start")

    data_ingestion = PythonOperator(
        task_id="data_ingestion",
        python_callable=ingest_data,
    )

    data_validation = PythonOperator(
        task_id="data_validation",
        python_callable=validate_data,
    )

    handle_missing = PythonOperator(
        task_id="handle_missing_values",
        python_callable=handle_missing_values,
    )

    feature_eng = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    data_encoding = PythonOperator(
        task_id="data_encoding",
        python_callable=encode_data,
    )

    model_training = PythonOperator(
        task_id="model_training",
        python_callable=train_model,
    )

    model_evaluation = PythonOperator(
        task_id="model_evaluation",
        python_callable=evaluate_model,
    )

    branching_decision = BranchPythonOperator(
        task_id="branching_decision",
        python_callable=choose_branch,
    )

    register_model = PythonOperator(
        task_id="register_model",
        python_callable=register_model_task,
    )

    reject_model = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model_task,
    )

    end = EmptyOperator(task_id="end")

    start >> data_ingestion >> data_validation
    data_validation >> [handle_missing, feature_eng]
    [handle_missing, feature_eng] >> data_encoding
    data_encoding >> model_training >> model_evaluation >> branching_decision
    branching_decision >> register_model >> end
    branching_decision >> reject_model >> end
