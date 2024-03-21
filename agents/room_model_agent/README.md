# room_model    

Ths component is responsible for creating a model of the robot's world. 
It subscribes to the *forcefield* component to get estimations of the room parameters.
The model is created using PyBullet.


## Configuration parameters
As any other component, *object_tracking* needs a configuration file to start. In
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
cd <object_tracking's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/object_tracking config
```
