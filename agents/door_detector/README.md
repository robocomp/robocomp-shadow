# door_detector_agent
Intro to component here

Install Clipper2 library : https://github.com/AngusJohnson/Clipper2

## Configuration parameters
As any other component, *door_detector_agent* needs a configuration file to start. In
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
cd <door_detector_agent's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/door_detector_agent config
```
