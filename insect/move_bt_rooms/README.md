# move_bt_rooms
It is necessary to run the component the BehaviorTree.cpp library, documentation is enable in official webpage https://www.behaviortree.dev
# Configuration parameters
As any other component, *move_bt_rooms* needs a configuration file to start. In
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
cd <move_bt_rooms's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/move_bt_rooms config
```
