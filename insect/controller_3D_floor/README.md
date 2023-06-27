# controller
Intro to component here

If regenerated you need to check that ui_mainUI.py has been generated with uic-qt6 (check in the header lines) 

Also in genericworker.py:
 - self.visualelements1_proxy = mprx["VisualElements1Proxy"]
 - from PySide6 import QtWidgets, QtCore


## Configuration parameters
As any other component, *controller* needs a configuration file to start. In
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
cd <controller's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/controller config
```
