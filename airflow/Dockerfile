FROM phusion/baseimage:0.10.1
RUN apt-get update && apt-get install -y python-setuptools python-pip python-dev libffi-dev libssl-dev zip wget
ENV SLUGIFY_USES_TEXT_UNIDECODE=yes
RUN pip install -U pip
# Airflow scheduler, by default this is sequential
# TODO: configure something like celery
RUN pip install apache-airflow
# initdb initializes a sqlite db for airflow bookeeping
# TODO: configure an actual db
# webserver starts a gunicorn/flask app for the airflow UI
CMD airflow initdb
# TODO: Figure out a way to exit if either webserver or scheduler crash
CMD airflow webserver -p 8080 -D
CMD airflow scheduler
