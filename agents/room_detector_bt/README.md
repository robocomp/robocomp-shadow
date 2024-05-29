# room_detector_bt
Intro to component here

## Instalation
To use *room_detector_bt* we need to install the BehaviorTree.CPP library. It is necessary to clone the repository and install the dependencies:

```
$ cd ~/software
$ git clone https://github.com/BehaviorTree/BehaviorTree.CPP/tree/master
$ cd BehaviorTree.CPP
$ sudo apt-get install libzqm3-dev libboost-dev
$ mkdir build ; cd build
$ make -j30
$ sudo make -j30 install
```
## CMake options
It is necessary to set the following CMake options to compile the component:
```
Add to behaviortree_cpp to LIBS section SET (LIBS ${LIBS})
Add the node.cpp (or the file where the BehaviorTrees nodes are implemented) file to the list of sources
```
## Configuration parameters
As any other component, *room_detector_bt* needs a configuration file to start. In
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
cd <room_detector_bt's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/room_detector_bt config
```
