# Description about how the methods for predicting approximate leakages were created

### 1. Approach
- Machine learning approach: generate feature vector X and feature vector Y:
- Feature vectors X contains:
  1. real sensor data for all 8 sensors
  2. another 8 features (one for each sensor) which means if the sensor was flagged as anomalous or not
    with other approaches before (matic anomaly detection), add this later, first just 8 features
  3. timestamp? optional, first just the 16 features
- Feature vector Y contains:
    - amount of leak
    - nodes that were in that group

 Try clustering approach that only keeps centroid and discard other data to save computation.   
 X: | timestamp | s1 | s2 | ... | s8 |     
 Y: | (node, leak), (node, leak), .... |    


## Suitable models
Train the model: either k-means, gaussian mixture, dbscan, or anything that precomputes centroids. 
Suitable implementations in sklearn:
- sklearn.cluster.MiniBatchKMeans
- sklearn.cluster.Birch
- ? ELKI's DBSCAN