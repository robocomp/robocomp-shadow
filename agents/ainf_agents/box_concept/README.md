# box_concept
Intro to component here

Box concept agent to detect, instantiate and maintain box concept instances in the DSR graph.


## Configuration parameters
As any other component, *box_concept* needs a configuration file to start. In
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
cd <box_concept's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/box_concept config
```
