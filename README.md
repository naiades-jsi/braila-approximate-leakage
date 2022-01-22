# Instructions

## Running the project
Dependencies needed to run the project:
- all packages in requirements.txt must be installed
- in directory ```data/divergence_matrix``` a pickle file 
  (named "Divergence_M.pickle" if you don't want to change the configuration) 
  must be present containing difference in pressure between values with no 
  leak and with leak
- kafka server must be running on the specified IP address and port


To run the project use Python3. Example command:    
```python3 main.py```

## Running as service on server
If you wish to run the project on the server as a service use the following:    
```pm2 start main.py --name braila_group_finder --interpreter python3```

More secure command so that we ensure that the service doesn't consume more memory than it should:
```pm2 start main.py --name braila_group_finder --interpreter python3 --max-memory-restart 7000M```

## Kafka related commands
Make sure that you are logged in as kafka user with proper permissions before running these commands.  

Command to get the list of all available topics:   
```~/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092```

Command to get latest messages (change the offset accordingly):    
```~/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic features_braila_leakage_detection --offset 1395 --partition 0```

Command to get all the messages since the beginning:   
```~/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic features_braila_leakage_detection --from-beginning```

Start a simple http server with pm2 and python:   
```pm2 start 'python3 -m http.server 8888' --name plotly_chart_serve```

Check for messages from this app:    
```~/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic braila_leakage_groups --from-beginning```

## Service output description

Below you see a sample output of the service.
```
{
    "timestamp": 1637239315,        
    "timestamp-processed-at": 1637239315,
    "critical-sensor": "SenzorComunarzi-NatVech",
    "deviation": 29.32,
    "method": "JNB",
    "epanet-file": "RaduNegru24May2021",
    "data": [
          {
            "node-name": "760-A",
            "latitude": 23.323,
            "longitude": 47.232,
            "group": 0
          },
          {
            "node-name": "Jonctiune-J-26",
            "latitude": 23.323,
            "longitude": 47.232,
            "group": 0
          },
          ...
    ]
}
```

Json keys above correspond to the following:
- `"timestamp"`: UNIX timestamp of when the data was captured on the sensor.
- `"timestamp-processed-at"`: UNIX timestamp of when the message was processed on the server.
- `"critical-sensor"`: Name of the sensor on which the leak was detected.
- `"deviation"`: Deviation from the simulated value.
- `"method"`: Method used to perform grouping of the nodes.
- `"epanet-file"`: Name/version of the EPANET file which was used to simulate the leak and generate node names.
- `"data"`: List of all nodes.
  - `"node-name"`: Name of the node corresponding to its name in EPANET.
  - `"latitude"`: Latitude of the node.
  - `"longitude"`: Longitude of the node.
  - `"group"`: Severity group of the node (0 being most severe).