FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY ./app /app

# RUN pip install -r ./requirements.txt
RUN apt-get clean \
    && apt-get -y update \
    && pip install --upgrade pip  \
    && apt-get -y install python3-dev \
    && apt-get -y install build-essential \
    && pip install -r requirements.txt \
    && rm -rf /var/cache/apk/*
