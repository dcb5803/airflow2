# Dockerfile
FROM apache/airflow:2.8.1-python3.9

USER root
RUN pip install --no-cache-dir pandas transformers

COPY airflow_llm_dag.py /opt/airflow/dags/
