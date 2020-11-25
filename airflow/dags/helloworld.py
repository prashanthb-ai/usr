# Apache airflow tutorial
# Source: https://airflow.apache.org/docs/stable/tutorial.html
# 3 steps:
# print_date prints date via `date` in bash
# sleep sleeps via `sleep` in bash
# templated


from datetime import timedelta

from airflow import DAG

from airflow.operators.bash_operator import BashOperator
from airflow.utils.dates import days_ago

# Args passed to all operators
default_args = {
        'owner': 'airflow',
        'depends_on_past': False,
        'start_date': days_ago(2),
        'email': ['airflow@example.com'],
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        }

dag = DAG(
        'helloworld',
        default_args=default_args,
        description='Hello world of apache airflow',
        schedule_interval=timedelta(days=1),
        )

t1 = BashOperator(
        task_id='print_date',
        bash_command='date',
        dag=dag,
        )

t2 = BashOperator(
        task_id='sleep',
        depends_on_past=False,
        bash_command='sleep 5',
        retries=3,
        dag=dag,
        )

dag.doc_md = __doc__

t1.doc_md = """"\
#### Task Documentation
You can document your task using the attributes `doc_md`, `doc`, `doc_rst`, `doc_json`, `doc_yaml` which gets rendered in the UI's Task Instance Details page.
![img](http://montcs.bloomu.edu/~bobmon/Semesters/2012-01/491/import%20soul.png)
"""

# Jinja template
# airflow macro: ds=date stamp, da_add adds days to ds
# params.my_param passed in via the BashOperator
templated_command = """
{% for i in range(5) %}
  echo "{{ ds }}"
  echo "{{ macros.ds_add(ds, 7) }}"
  echo "{{ params.my_param }}"
{% endfor %}
"""

t3 = BashOperator(
        task_id='templated',
        depends_on_past=False,
        bash_command=templated_command,
        params={'my_param': 'Test Parameter'},
        dag=dag,
        )

t1 >> [t2, t3]
