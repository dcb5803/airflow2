# airflow_llm_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
from transformers import pipeline

def load_dataset():
    # Load a small dataset inline (can be replaced with external source)
    data = pd.DataFrame({
        "text": [
            "Airflow orchestrates workflows efficiently.",
            "Transformers are powerful for NLP tasks.",
            "Docker simplifies deployment."
        ]
    })
    data.to_csv("/tmp/input.csv", index=False)

def run_llm():
    # Load dataset
    df = pd.read_csv("/tmp/input.csv")
    # Load open-source LLM pipeline
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    # Run summarization
    df["summary"] = df["text"].apply(lambda x: summarizer(x, max_length=30, min_length=5, do_sample=False)[0]["summary_text"])
    df.to_csv("/tmp/output.csv", index=False)

with DAG(
    dag_id="airflow_llm_demo",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["llm", "airflow", "demo"]
) as dag:

    task_load = PythonOperator(
        task_id="load_dataset",
        python_callable=load_dataset
    )

    task_llm = PythonOperator(
        task_id="run_llm",
        python_callable=run_llm
    )

    task_load >> task_llm
