import pickle
from pathlib import Path

from airflow import DAG
from random import randint
from datetime import datetime
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator

import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error

import mlflow

mlflow_tracking_uri = "http://mlflow-server:5000"
mlflow.set_tracking_uri(mlflow_tracking_uri)
mlflow.set_experiment("nyc-taxi-experiment")


def _read_dataframe(year, month):
	url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_{year}-{month:02d}.parquet'
	df = pd.read_parquet(url)

	df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
	df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

	df = df[(df.duration >= 1) & (df.duration <= 60)]

	categorical = ['PULocationID', 'DOLocationID']
	df[categorical] = df[categorical].astype(str)

	df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

	return df

def _create_dicts(ti, upstream_task_id: str):
	df = ti.xcom_pull(task_ids=upstream_task_id)
	target = 'duration'
	y = df[target].values
	
	categorical = ['PU_DO']
	numerical = ['trip_distance']
	dicts = df[categorical + numerical].to_dict(orient='records')
	
	return (dicts, y.tolist())

def _train_model(ti):
	dicts_train, y_train = ti.xcom_pull(task_ids='create_dict_train')
	dicts_val, y_val = ti.xcom_pull(task_ids='create_dict_val')

	dv = DictVectorizer(sparse=True)
	X_train = dv.fit_transform(dicts_train)
	X_val = dv.transform(dicts_val)
	
	y_train = np.array(y_train)
	y_val = np.array(y_val)

	local_models_path = Path("models")
	local_models_path.mkdir(exist_ok=True)
	preprocessor_path = local_models_path / "preprocessor.b"

	with mlflow.start_run() as run:
		train = xgb.DMatrix(X_train, label=y_train)
		valid = xgb.DMatrix(X_val, label=y_val)
		
		best_params = {
			'learning_rate': 0.09585355369315604,
			'max_depth': 30,
			'min_child_weight': 1.060597050922164,
			'objective': 'reg:linear',
			'reg_alpha': 0.018060244040060163,
			'reg_lambda': 0.011658731377413597,
			'seed': 42
			}
		
		mlflow.log_params(best_params)
		
		booster = xgb.train(
			params=best_params,
			dtrain=train,
			num_boost_round=30,
			evals=[(valid, 'validation')],
			early_stopping_rounds=50
		)

		y_pred = booster.predict(valid)
		rmse = root_mean_squared_error(y_val, y_pred)
		mlflow.log_metric("rmse", rmse)

		with open(preprocessor_path, "wb") as f_out:
			pickle.dump(dv, f_out)
		mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")

		mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

		return run.info.run_id


def _run(ti):

	run_id = ti.xcom_pull(task_ids='train_model')
	print(f"MLflow run_id: {run_id}")
	return run_id

with DAG("duration_prediction_dag", start_date=datetime(2025, 6, 5), schedule="@monthly", catchup=False) as dag:
	year = 2021
	month = 1

	read_df_train = PythonOperator(
		task_id="read_df_train",
		python_callable=_read_dataframe,
		op_kwargs={"year": year, "month": month}
	)

	read_df_val = PythonOperator(
		task_id="read_df_val",
		python_callable=_read_dataframe,
		op_kwargs={"year": year, "month": month + 1}
	)

	create_dict_train = PythonOperator(
		task_id="create_dict_train",
		python_callable=_create_dicts,
		op_kwargs={"upstream_task_id": "read_df_train"},  # Pass upstream task ID
	)

	create_dict_val = PythonOperator(
		task_id="create_dict_val",
		python_callable=_create_dicts,
		op_kwargs={"upstream_task_id": "read_df_val"},  # Pass upstream task ID
	)

	train_model = PythonOperator(
		task_id="train_model",
		python_callable=_train_model,
	)

	run = PythonOperator(
		task_id="run",
		python_callable=_run,
	)

read_df_train >> create_dict_train 
read_df_val >> create_dict_val
[create_dict_train, create_dict_val] >> train_model >> run