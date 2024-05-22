# long_term_spatial_memory_agent
This agent manages the long term spatial memory of the CORTEX architecture.
It does the following functions:

- remove rooms and objects in it not inhabited by the robot from the DWM and store them internally 
- add rooms that are to be entered by the robot
- answer to queries for paths connecting two locations
- answer to queries for the location of an object 

## Configuration parameters
As any other component, *long_term_spatial_memory_agent* needs a configuration file to start. In
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
cd <long_term_spatial_memory_agent's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/long_term_spatial_memory_agent config
```
