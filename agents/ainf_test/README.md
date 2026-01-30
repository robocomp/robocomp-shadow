# ainf_test
Intro to component here

Test component to validate Active Inference in the room-robot concept problem
Ejecutar el script python subcognitive.py sub.toml
Cargar el mundo de webots: SimpleWorld.wbt

## Configuration parameters
As any other component, *ainf_test* needs a configuration file to start. In
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
cd <ainf_test's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/ainf_test config
```
