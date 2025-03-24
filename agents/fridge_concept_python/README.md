# fridge_concept_python
Intro to component here

Agent for CORTEX that maintains the knowledge of a fridge concept. 
It receive lidar data an decides if new instance of the concept are created
It maintains the created instances and updates them with new data

## Installation

ATTENTION: In Ubuntu 22.04 with Python 3.10, it requires the creation of a venv with Python3.11

python3.11 -m venv .venv
source .venv/bin/activate
uv pip install torch, rich, PySide6, zeroc-ice, numpy, vispy, shapely, open3d
```

## Configuration parameters
As any other component, *fridge_concept_python* needs a configuration file to start. In
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
cd <fridge_concept_python's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/fridge_concept_python config
```
