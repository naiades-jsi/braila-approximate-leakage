FROM bitnami/minideb:bullseye

RUN apt-get -y update
RUN apt-get install -y build-essential \
                       python3-dev \
                       python3-pip

RUN mkdir '/app'
WORKDIR '/app'
ADD . .

# prepare internal directory for images
# this directory is ignored in .dockerignore to avoid transferring existing images
# into docker
RUN mkdir 'grafana-files'

RUN pip3 install -r requirements.txt

# prepare the logs directory
RUN mkdir -p logs

WORKDIR '/app/bin'
CMD ["./run-release"]