## Divergence matrix

This folder should contain pickle files provided by our partners which 
are later used in the application.

### Data format of the pickle file

The pickle file should contain the data in the following format:   
```
[pandas.DataFrame, pandas.DataFrame,pandas.DataFrame, ...]
``` 

Each dataframe stores in its columns the names of the nodes in the EPANET network
and as its rows second of the day for which the data was simulated. Example of the 
dataframe can be seen below:

|  | Node_1 | Node_2 | Node_3 | ... | Node_n| 
|---|---|---|---|---|---|
| 3600   | 0.11    | 0.12    | 0.09    | ... | 0.11  | 
| 7200   | 0.11    | 0.11    | 0.11    | ... | 0.11  | 
| 10800  | 0.11    | 0.11    | 0.11    | ... | 0.11  |  
| ...    | ...     | ...     | ...     | ... | ...   | 
| 86400  | 0.11    | 0.11    | 0.11    | ... | 0.11  | 


The dataframe should also have a name (dataframe.columns.name) which contains 
the data about on which node the leak was placed and the amount of leak in LPS.    
Example of a name in the correct format: ```'Node_Jonctiune-267, 4.0LPS'```