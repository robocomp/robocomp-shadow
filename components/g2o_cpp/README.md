# g2o_cpp
This component is a wrapper for the g2o library. It is used to perform graph optimization on a graph of nodes and edges. The nodes represent the poses of the robot and the edges represent the constraints between the poses. 
The constraints can be odometry measurements, loop closures, etc.
The component implements the G2Ooptimizer.idsl interface, which allows the user to send a string with the graph in the g2o format and receive the optimized graph in the same format.


## Configuration parameters
As any other component, *g2o_cpp* needs a configuration file to start. In
```
etc/config
```
you can find an example of a configuration file. We can find there the following lines:
```
EXAMPLE HERE
```

## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

```
cd <g2o_cpp's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/g2o_cpp config
```
