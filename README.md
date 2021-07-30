# Instructions

## Running the project
To run the project use Python3. Example command:    
```python3 main.py```

If you wish to run the project on Atena as a service use the following:    
```pm2 start main.py --name braila_group_finder --interpreter python3```


## Kafka related command
Make sure that you are logged in as kafka user with proper permissions before running these commands.  

Command to get the list of all available topics:   
```~/kafka/bin/kafka-topics.sh --list --bootstrap-server localhost:9092```

Command to get latest messages (change the offset accordingly):    
```~/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic features_braila_leakage_detection --offset 1395 --partition 0```

Command to get all the messages since the beginning:   
```~/kafka/bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic features_braila_leakage_detection --from-beginning```