#!/bin/bash

data_dir="/home/lstopar/data/naiadas/neo4j"


docker run \
    --name testneo4j \
    -p7474:7474 -p7687:7687 \
    -d \
    -v $data_dir/neo4j/data:/data \
    -v $data_dir/neo4j/logs:/logs \
    -v $data_dir/neo4j/import:/var/lib/neo4j/import \
    -v $data_dir/neo4j/plugins:/plugins \
    --env NEO4J_AUTH=neo4j/test \
    neo4j:latest
