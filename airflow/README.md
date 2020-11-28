# Apache Airflow

To install airflow manually, follow the steps in the Dockerfile. To start airflow manually:
```
$ airflow webserver -p 8080 -D
$ airflow scheduler
```

Container usage
```
$ docker build . -t bprashanth/apache-airflow:0.1
$ docker run -p 8080:8080 bprashanth/apache-airflow:0.1
```

The `airflow` python library makes certain assumptions about the port and
directory structure when it starts up.

## Configuration

* Default config and db are stored in `~/airflow`
* Note that this has a `dags_folder` variable pointing at the folder with
  configuration of all DAGs (the python scripts you generate/write)
* This also has variables for the `executor` (`Sequential`), webserver and
  connection pools
```
$ airflow list_dags

-------------------------------------------------------------------
DAGS
-------------------------------------------------------------------
example_bash_operator
helloworld
tutorial
...
$ python ./dags/helloworld.py
...
$ airflow list_tasks helloworld
```

Running DAGs
```
$ airflow trigger_dag helloworld
```

* Trigger dags work even if the DAG is "off" in the UI
* Turning the DAG "on" in the UI is equivalent to running `airflow run`

## Debug

* Scripts showing up in `list_dags` but not in the UI
Modify the dags directory in `$AIRFLOW_HOME/airflow.cfg` and run the scheduler
```
$ airflow scheduler
```

* Scheduler crashes with permission denied from the logs dir listed in `airflow.cfg`
```
  [Previous line repeated 5 more times]
  File "/home/radifar/.virtualenvs/airflow/lib/python3.7/os.py", line 221, in makedirs
    mkdir(name, mode)
PermissionError: [Errno 13] Permission denied: '/media/radifar/radifar-dsl/Workflow/Airflow/airflow-home/logs/scheduler/2020-01-04/../../../../../../../home'
```

* set `load_examples=False` in `airflow.cfg`
    - probably a bug, it's using a relative link from the specified DAG dir for
      the logs dir too (which is configured as `~/airflow/logs/`)

## Appendix

* [Quick start](https://airflow.apache.org/docs/stable/start.html)
* [Cloud compose](https://cloud.google.com/composer/)
