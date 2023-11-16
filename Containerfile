FROM registry.access.redhat.com/ubi9/python-39

ARG MYSQL_HOST
ARG MYSQL_PORT

ENV MYSQL_HOST ${MYSQL_HOST}
ENV MYSQL_PORT ${MYSQL_PORT}

USER 0
#WORKDIR /tmp/src

ADD src/requirements.txt .
RUN pip install -r requirements.txt

ADD . .
# RUN python load_dataset.py
# RUN python sample_dataset.py
# RUN python training.py
# # deberia dejar un pickle en ./model.pkl
# RUN python scoring.py

ENTRYPOINT [ "/bin/bash", "./src/run_challenge.sh"]

