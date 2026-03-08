"""
MLOps Airflow + MLflow Pipeline — Titanic Survival Prediction
==============================================================
DAG that orchestrates the full ML lifecycle:
  data ingestion → validation → parallel preprocessing →
  encoding → model training (MLflow) → evaluation →
  branching (register / reject model).

Trigger with different hyperparameters via dag_run.conf, e.g.:
  {"model_type": "random_forest", "n_estimators": 100, "max_depth": 5}
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timedelta

import mlflow
import pandas as pd
from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = os.environ.get(
    "TITANIC_DATASET_PATH",
    "/Users/ahmad/Library/CloudStorage/OneDrive-Personal/University/Semester-8/MLOPs/Assignment-2/Titanic-Dataset.csv",
)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "Titanic_Survival_Prediction"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default DAG arguments
# ---------------------------------------------------------------------------
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(seconds=30),
}

# =========================================================================
#  TASK FUNCTIONS
# =========================================================================

# ---- Task 2: Data Ingestion -------------------------------------------
def data_ingestion(**context):
    """Load Titanic CSV, print shape, log missing values, push path via XCom."""
    df = pd.read_csv(DATASET_PATH)

    # Print dataset shape
    log.info("Dataset shape: %s", df.shape)
    print(f"Dataset shape: {df.shape}")

    # Log missing values count
    missing = df.isnull().sum()
    log.info("Missing values per column:\n%s", missing.to_string())
    print(f"Missing values:\n{missing}")

    # Push dataset path via XCom
    context["ti"].xcom_push(key="dataset_path", value=DATASET_PATH)
    return DATASET_PATH


# ---- Task 3: Data Validation ------------------------------------------
def data_validation(**context):
    """
    Check missing % in Age and Embarked.
    Raise exception if missing > 30%.
    This task has retries=2 to demonstrate retry behaviour.
    """
    dataset_path = context["ti"].xcom_pull(
        task_ids="data_ingestion", key="dataset_path"
    )
    df = pd.read_csv(dataset_path)

    total_rows = len(df)
    age_missing_pct = df["Age"].isnull().sum() / total_rows * 100
    embarked_missing_pct = df["Embarked"].isnull().sum() / total_rows * 100

    log.info("Age missing: %.2f%%", age_missing_pct)
    log.info("Embarked missing: %.2f%%", embarked_missing_pct)
    print(f"Age missing: {age_missing_pct:.2f}%")
    print(f"Embarked missing: {embarked_missing_pct:.2f}%")

    # --- Intentional failure demo (set Airflow Variable
    #     'force_validation_failure' to 'true' to trigger) ---
    from airflow.models import Variable

    force_fail = Variable.get("force_validation_failure", default_var="false")
    if force_fail.lower() == "true":
        # Reset the variable so subsequent retries pass
        Variable.set("force_validation_failure", "false")
        raise ValueError(
            "INTENTIONAL FAILURE for retry demo — "
            "variable 'force_validation_failure' was set to true."
        )

    # Actual validation
    if age_missing_pct > 30:
        raise ValueError(
            f"Age column has {age_missing_pct:.2f}% missing values (> 30%)"
        )
    if embarked_missing_pct > 30:
        raise ValueError(
            f"Embarked column has {embarked_missing_pct:.2f}% missing values (> 30%)"
        )

    log.info("Data validation PASSED.")
    return "validation_passed"


# ---- Task 4a: Handle Missing Values -----------------------------------
def handle_missing_values(**context):
    """Fill missing Age with median and Embarked with mode."""
    dataset_path = context["ti"].xcom_pull(
        task_ids="data_ingestion", key="dataset_path"
    )
    df = pd.read_csv(dataset_path)

    df["Age"].fillna(df["Age"].median(), inplace=True)
    df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

    # Save processed data to a temp file and push its path
    tmp_path = os.path.join(tempfile.gettempdir(), "titanic_missing_handled.csv")
    df.to_csv(tmp_path, index=False)
    context["ti"].xcom_push(key="missing_handled_path", value=tmp_path)
    log.info("Missing values handled. Saved to %s", tmp_path)
    return tmp_path


# ---- Task 4b: Feature Engineering -------------------------------------
def feature_engineering(**context):
    """Create FamilySize and IsAlone features."""
    dataset_path = context["ti"].xcom_pull(
        task_ids="data_ingestion", key="dataset_path"
    )
    df = pd.read_csv(dataset_path)

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    tmp_path = os.path.join(tempfile.gettempdir(), "titanic_features.csv")
    df.to_csv(tmp_path, index=False)
    context["ti"].xcom_push(key="features_path", value=tmp_path)
    log.info("Feature engineering done. Saved to %s", tmp_path)
    return tmp_path


# ---- Task 5: Data Encoding --------------------------------------------
def data_encoding(**context):
    """
    Merge the outputs of parallel tasks, encode categoricals, drop irrelevant cols.
    """
    missing_path = context["ti"].xcom_pull(
        task_ids="handle_missing_values", key="missing_handled_path"
    )
    features_path = context["ti"].xcom_pull(
        task_ids="feature_engineering", key="features_path"
    )

    df_missing = pd.read_csv(missing_path)
    df_features = pd.read_csv(features_path)

    # Merge: take imputed columns from df_missing, new features from df_features
    df = df_missing.copy()
    df["FamilySize"] = df_features["FamilySize"]
    df["IsAlone"] = df_features["IsAlone"]

    # Encode Sex: male=1, female=0
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # One-hot encode Embarked
    embarked_dummies = pd.get_dummies(df["Embarked"], prefix="Embarked", dtype=int)
    df = pd.concat([df, embarked_dummies], axis=1)

    # Drop irrelevant columns
    cols_to_drop = ["Name", "Ticket", "Cabin", "PassengerId", "Embarked"]
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

    tmp_path = os.path.join(tempfile.gettempdir(), "titanic_encoded.csv")
    df.to_csv(tmp_path, index=False)
    context["ti"].xcom_push(key="encoded_path", value=tmp_path)
    log.info("Data encoding done. Shape: %s", df.shape)
    return tmp_path


# ---- Task 6: Model Training with MLflow --------------------------------
def model_training(**context):
    """Train model, log hyperparams and artifacts to MLflow."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    encoded_path = context["ti"].xcom_pull(
        task_ids="data_encoding", key="encoded_path"
    )
    df = pd.read_csv(encoded_path)

    # Get hyperparameters from dag_run.conf (or defaults)
    conf = context.get("dag_run").conf or {}
    model_type = conf.get("model_type", "logistic_regression")
    lr_C = float(conf.get("C", 1.0))
    lr_max_iter = int(conf.get("max_iter", 200))
    rf_n_estimators = int(conf.get("n_estimators", 100))
    rf_max_depth = conf.get("max_depth", None)
    if rf_max_depth is not None:
        rf_max_depth = int(rf_max_depth)

    # Train/test split
    X = df.drop(columns=["Survived"])
    y = df["Survived"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        if model_type == "random_forest":
            mlflow.log_param("n_estimators", rf_n_estimators)
            mlflow.log_param("max_depth", rf_max_depth)
            model = RandomForestClassifier(
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                random_state=42,
            )
        else:
            mlflow.log_param("C", lr_C)
            mlflow.log_param("max_iter", lr_max_iter)
            model = LogisticRegression(C=lr_C, max_iter=lr_max_iter, random_state=42)

        model.fit(X_train, y_train)

        # Log model artifact
        mlflow.sklearn.log_model(model, artifact_path="model")

        log.info("Model training complete. Run ID: %s", run_id)

    # Save test data for evaluation
    test_path = os.path.join(tempfile.gettempdir(), "titanic_test.csv")
    test_df = X_test.copy()
    test_df["Survived"] = y_test
    test_df.to_csv(test_path, index=False)

    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="test_path", value=test_path)
    context["ti"].xcom_push(key="model_type", value=model_type)
    return run_id


# ---- Task 7: Model Evaluation -----------------------------------------
def model_evaluation(**context):
    """Compute metrics, log to MLflow, push accuracy via XCom."""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    run_id = context["ti"].xcom_pull(task_ids="model_training", key="run_id")
    test_path = context["ti"].xcom_pull(task_ids="model_training", key="test_path")
    model_type = context["ti"].xcom_pull(task_ids="model_training", key="model_type")

    test_df = pd.read_csv(test_path)
    y_test = test_df["Survived"]
    X_test = test_df.drop(columns=["Survived"])

    # Load the logged model
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    log.info("Accuracy:  %.4f", accuracy)
    log.info("Precision: %.4f", precision)
    log.info("Recall:    %.4f", recall)
    log.info("F1-score:  %.4f", f1)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Log metrics back to the same MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    # Push accuracy via XCom
    context["ti"].xcom_push(key="accuracy", value=accuracy)
    context["ti"].xcom_push(key="run_id", value=run_id)
    context["ti"].xcom_push(key="model_type", value=model_type)
    return accuracy


# ---- Task 8: Branching Logic ------------------------------------------
def check_accuracy(**context):
    """BranchPythonOperator callable: route based on accuracy threshold."""
    accuracy = context["ti"].xcom_pull(task_ids="model_evaluation", key="accuracy")
    log.info("Accuracy = %.4f — threshold is 0.80", accuracy)

    if accuracy >= 0.80:
        return "register_model"
    else:
        return "reject_model"


# ---- Task 9a: Register Model ------------------------------------------
def register_model(**context):
    """Register the model in MLflow Model Registry."""
    run_id = context["ti"].xcom_pull(task_ids="model_evaluation", key="run_id")
    model_type = context["ti"].xcom_pull(task_ids="model_evaluation", key="model_type")
    accuracy = context["ti"].xcom_pull(task_ids="model_evaluation", key="accuracy")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    model_name = "TitanicSurvivalModel"

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    log.info(
        "Model registered: name=%s, version=%s (accuracy=%.4f, type=%s)",
        result.name,
        result.version,
        accuracy,
        model_type,
    )
    print(
        f"✅ Model REGISTERED — {model_name} v{result.version} "
        f"(accuracy={accuracy:.4f}, type={model_type})"
    )


# ---- Task 9b: Reject Model --------------------------------------------
def reject_model(**context):
    """Log rejection reason (accuracy too low) to MLflow."""
    run_id = context["ti"].xcom_pull(task_ids="model_evaluation", key="run_id")
    accuracy = context["ti"].xcom_pull(task_ids="model_evaluation", key="accuracy")
    model_type = context["ti"].xcom_pull(task_ids="model_evaluation", key="model_type")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("registration_status", "REJECTED")
        mlflow.log_param(
            "rejection_reason",
            f"Accuracy {accuracy:.4f} < 0.80 threshold",
        )

    log.info(
        "Model REJECTED: accuracy=%.4f, type=%s", accuracy, model_type
    )
    print(
        f"❌ Model REJECTED — accuracy={accuracy:.4f} < 0.80 (type={model_type})"
    )


# =========================================================================
#  DAG DEFINITION  (Task 1)
# =========================================================================
with DAG(
    dag_id="mlops_pipeline",
    default_args=default_args,
    description="End-to-end ML pipeline: Titanic survival prediction with MLflow",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops", "titanic", "mlflow"],
) as dag:

    # --- Task 2 ---
    t_ingest = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )

    # --- Task 3 (retries=2 to demonstrate retry behaviour) ---
    t_validate = PythonOperator(
        task_id="data_validation",
        python_callable=data_validation,
        retries=2,
        retry_delay=timedelta(seconds=30),
    )

    # --- Task 4: Parallel processing ---
    t_missing = PythonOperator(
        task_id="handle_missing_values",
        python_callable=handle_missing_values,
    )

    t_features = PythonOperator(
        task_id="feature_engineering",
        python_callable=feature_engineering,
    )

    # --- Task 5 ---
    t_encoding = PythonOperator(
        task_id="data_encoding",
        python_callable=data_encoding,
    )

    # --- Task 6 ---
    t_train = PythonOperator(
        task_id="model_training",
        python_callable=model_training,
    )

    # --- Task 7 ---
    t_evaluate = PythonOperator(
        task_id="model_evaluation",
        python_callable=model_evaluation,
    )

    # --- Task 8: BranchPythonOperator ---
    t_branch = BranchPythonOperator(
        task_id="check_accuracy",
        python_callable=check_accuracy,
    )

    # --- Task 9 ---
    t_register = PythonOperator(
        task_id="register_model",
        python_callable=register_model,
    )

    t_reject = PythonOperator(
        task_id="reject_model",
        python_callable=reject_model,
    )

    # ---- Dependencies (no cyclic dependencies) ----
    #
    #  data_ingestion >> data_validation >> [handle_missing, feature_engineering]
    #  >> data_encoding >> model_training >> model_evaluation >> check_accuracy
    #  >> [register_model | reject_model]
    #
    t_ingest >> t_validate
    t_validate >> [t_missing, t_features]          # Parallel (Task 4)
    [t_missing, t_features] >> t_encoding          # Join
    t_encoding >> t_train >> t_evaluate >> t_branch
    t_branch >> [t_register, t_reject]             # Branch (Task 8)
