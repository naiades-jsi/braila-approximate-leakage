
## Information about the sensors: 

1. Sensors:
- urn:ngsi-ld:Device:Device-5771 (address: Strada Castanului 84),
- urn:ngsi-ld:Device:Device-5772 (address: Strada Chișinău 44),
- urn:ngsi-ld:Device:Device-5770 (address: Strada Comunarzi 2),
- urn:ngsi-ld:Device:Device-5773 (address: Strada Sebeșului 132)   
Kafka topics:
- measurements_node_braila_pressure5771
- measurements_node_braila_pressure5772
- measurements_node_braila_pressure5770
- measurements_node_braila_pressure5773
- measurements_node_braila_pressure5770_night (samples with timestamps between 23h and 1h)
- measurements_node_braila_pressure5771_night (samples with timestamps between 23h and 1h)
- measurements_node_braila_pressure5772_night (samples with timestamps between 23h and 1h)
- measurements_node_braila_pressure5773_night (samples with timestamps between 23h and 1h)

2. Debitmeters:
- urn:ngsi-ld:Device:Device-318505H498,
- urn:ngsi-ld:Device:Device-211306H360,
- urn:ngsi-ld:Device:Device-211106H360,
- urn:ngsi-ld:Device:Device-211206H360  
Kafka topics:
- measurement_node_braila_flow318505H498,
- measurement_node_braila_flow211306H360,
- measurement_node_braila_flow211106H360,
- measurement_node_braila_flow211206H360  
Values:
- flow_rate_value: ?
- totalizer1: ?
- totalizer2: ?
- consumer_totalizer?
- analog_input1: ?
- analog_input2: ?
- batery_capacity: ?
- alarms_in_decimal: ?

## From emails

### analog2
The debitmeters/flowmeters provide pressure through the Analog 2 column, which registers a voltage change
depending on pressure. The number there is the raw voltage data that the debitmeter has recorded.

From that, to get the actual pressure data, you need to apply a formula of "(analoge-0.6)*4".

This is how the Siemens equipment functions. Sorry if it wasn't made clear to you previously.
If the value you get is negative, then it means the pressure sensor was non-functioning for that period of time, or
that the flow on that section of pipe was shut down for that period.

### Debitmeters
The flow meters (demand sensors)map to the following names/coordinates/junctions:
  

| Name  | Coordinates  | ID  | Node  |  
|---|---|---|---|
| Apollo  | 45.25273362, 27.93552109  | 211206H360 |  Jonctiune-3974   | 
| GA-Braila  | 45.25228252, 27.94718523  |  477105H429 (formally 211106H360) | Jonctiune-J-3  | 
| RaduNegru 2 | 45.24572269, 27.94187308  | 318505H498  | Jonctiune-J-19   | 
| RaduNegruMare  | 45.23905319, 27.93228389  | 211306H360  | Jonctiune-2749  | 


The reservoirs and pumps present in the model do not exist in real life.
They are there to simulate the water coming into the network through the junctions at the nodes in the table above.
Our actual network has three main pumping stations located far away from this DMA, and feed the entire city.
The data used for the virtual pumps that are in the model, connected to the junctions from the table above, is taken from the flow meter data and tweaked a bit so that the model will run.
The reservoirs are just there so that there would be something for the system to draw water from.