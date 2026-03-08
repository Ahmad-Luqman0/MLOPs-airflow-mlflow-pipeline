# MLOps Pipeline — Titanic Survival Prediction

End-to-end ML pipeline using **Apache Airflow** (DAG orchestration) and **MLflow** (experiment tracking & model registry).

## Project Structure

```
Assignment-2/
├── dags/
│   └── mlops_airflow_mlflow_pipeline.py   # Airflow DAG
├── Titanic-Dataset.csv                     # Dataset
├── requirements.txt                        # Python dependencies
└── README.md
```

## DAG Overview

```
data_ingestion → data_validation (retries=2)
                       ↓
          ┌────────────┴────────────┐
          ↓                         ↓        ← Parallel
   handle_missing_values    feature_engineering
          └────────────┬────────────┘
                       ↓
               data_encoding
                       ↓
            model_training (MLflow)
                       ↓
            model_evaluation
                       ↓
            check_accuracy (Branch)
               ↙            ↘
       register_model    reject_model
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize Airflow

```bash
export AIRFLOW_HOME=~/airflow

# Initialize the database
airflow db migrate

# Create admin user
airflow users create \
    --username admin \
    --password admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com
```

### 3. Deploy the DAG

```bash
mkdir -p $AIRFLOW_HOME/dags
cp dags/mlops_airflow_mlflow_pipeline.py $AIRFLOW_HOME/dags/
```

### 4. Start MLflow Tracking Server

```bash
mlflow server --host 127.0.0.1 --port 5000
```

### 5. Start Airflow

```bash
# In separate terminals:
airflow webserver --port 8080
airflow scheduler
```

### 6. Access UIs

- **Airflow**: http://localhost:8080 (user: `admin`, pass: `admin`)
- **MLflow**: http://127.0.0.1:5000

## Running the Pipeline

### Run 1 — Logistic Regression

```bash
airflow dags trigger mlops_pipeline \
    --conf '{"model_type":"logistic_regression","C":1.0,"max_iter":200}'
```

### Run 2 — Random Forest (shallow)

```bash
airflow dags trigger mlops_pipeline \
    --conf '{"model_type":"random_forest","n_estimators":100,"max_depth":5}'
```

### Run 3 — Random Forest (deep)

```bash
airflow dags trigger mlops_pipeline \
    --conf '{"model_type":"random_forest","n_estimators":200,"max_depth":10}'
```

## Demonstrating Retry Behaviour

To trigger an intentional failure in the `data_validation` task (to demonstrate retries):

```bash
airflow variables set force_validation_failure true
```

Then trigger the DAG. The first attempt will fail, Airflow will retry (up to 2 times), and the second attempt will succeed because the variable gets reset to `false`.

## Experiment Comparison

After running the DAG 3 times with different hyperparameters, open the MLflow UI and compare runs side-by-side to identify the best-performing model based on accuracy, precision, recall, and F1-score.
