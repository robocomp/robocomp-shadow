# Installation instructions
- sudo apt-get install libsuitesparse-dev
- Follow instructions in "Installation from source" section in https://github.com/miquelmassot/g2opy/tree/master

 # DOES NOT WORK WITH NUMPY 2

# g2o_agent
Intro to component here


## Configuration parameters
As any other component, *g2o_agent* needs a configuration file to start. In
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
cd <g2o_agent's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/g2o_agent config
```
