# BUILDING: docker build -t <container_name> .
# RUNNING: docker run <args> <container_name>
# e.g. docker run -d --network="host" leakage_approximate
FROM ubuntu:20.04
RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev
COPY ./requirements.txt /requirements.txt
WORKDIR /
RUN pip3 install -r requirements.txt
COPY . /
CMD ["python3", "main_pretrained_models.py"]
