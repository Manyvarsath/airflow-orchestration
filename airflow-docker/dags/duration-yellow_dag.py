import pickle
from pathlib import Path

from airflow import DAG
from random import randint
from datetime import datetime
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
import logging

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow_tracking_uri = "http://mlflow-server:5000"
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("yellow-march-taxi-duration")

log = logging.getLogger(__name__)

def _read_dataframe():
	url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet"
	df = pd.read_parquet(url)
	log.info(f"Length of the DataFrame before filtering: {len(df)}")
	
	df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
	df.duration = df.duration.dt.total_seconds() / 60	
	
	df = df[(df.duration >= 1) & (df.duration <= 60)]	
	
	categorical = ['PULocationID', 'DOLocationID']
	df[categorical] = df[categorical].astype(str)
	log.info(f"Length of the DataFrame after filtering: {len(df)}")

	temp_data_dir = Path("temp_airflow_data")
	temp_data_dir.mkdir(parents=True, exist_ok=True)
	output_file_path = temp_data_dir / "march_2023_yellow.parquet"
	df.to_parquet(output_file_path)
	return str(output_file_path)

def _create_dicts(ti):
	tmp_path = ti.xcom_pull(task_ids="read_df")
	df = pd.read_parquet(tmp_path)
	log.info(f"DataFrame loaded from {tmp_path}")

	target = 'duration'
	y = df[target].values

	log.info(f"Length of the DataFrame after filtering: {len(df)}")
	
	categorical = ['PULocationID', 'DOLocationID']
	dicts = df[categorical].to_dict(orient='records')

	temp_data_dir = Path("temp_airflow_data")
	temp_data_dir.mkdir(parents=True, exist_ok=True)

	dicts_file_path = temp_data_dir / "feature_dicts.pkl"
	y_list_file_path = temp_data_dir / "target_list.pkl"

	with open(dicts_file_path, "wb") as f_out:
		pickle.dump(dicts, f_out)
	log.info(f"Feature dictionaries saved to {dicts_file_path}")

	with open(y_list_file_path, "wb") as f_out:
		pickle.dump(y.tolist(), f_out)
	
	return (str(dicts_file_path), str(y_list_file_path))

def _train_model(ti):
	dicts_file_path_str, y_list_file_path_str = ti.xcom_pull(task_ids='create_dict')
	with open(dicts_file_path_str, "rb") as f_in:
		dicts_train = pickle.load(f_in)
	log.info(f"Feature dictionaries loaded from {dicts_file_path_str}")

	with open(y_list_file_path_str, "rb") as f_in:
		y_list_train = pickle.load(f_in)
	log.info(f"Target list loaded from {y_list_file_path_str}")

	dv = DictVectorizer()
	X_train = dv.fit_transform(dicts_train)
	y_train = np.array(y_list_train)

	local_models_path = Path("models")
	local_models_path.mkdir(exist_ok=True)
	preprocessor_path = local_models_path / "preprocessor.b"

	with mlflow.start_run() as run:
		lr = LinearRegression()
		lr.fit(X_train, y_train)

		y_pred_train = lr.predict(X_train)

		rmse_train = root_mean_squared_error(y_train, y_pred_train)
		mlflow.log_metric("Training data rmse", rmse_train)
		mlflow.log_metric("intercept", lr.intercept_)

		log.info(f"Model intercept: {lr.intercept_}")

		with open(preprocessor_path, "wb") as f_out:
			pickle.dump(dv, f_out)
		mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

		mlflow.sklearn.log_model(lr, artifact_path="model", registered_model_name="TaxiDurationModel")
		mlflow.log_param("Vectoriezer", dv)
	return run.info.run_id


with DAG("duration_yellow_prediction_dag", start_date=datetime(2025, 6, 5), schedule="@monthly", catchup=False) as dag:

	read_df = PythonOperator(
			task_id="read_df",
			python_callable=_read_dataframe
		)

	create_dict = PythonOperator(
			task_id="create_dict",
			python_callable=_create_dicts
			)

	train_model = PythonOperator(
			task_id="train_model",
			python_callable=_train_model,
		)

read_df >> create_dict >> train_model