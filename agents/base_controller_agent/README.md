# base_controller_agent
This agent is responsible for controlling the robot's base. It detects an [intention] edge in G and tries to bring the robot to the corresponding object.


## Configuration parameters
As any other component, *base_controller_agent* needs a configuration file to start. In
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
cd <base_controller_agent's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/base_controller_agent config
```
