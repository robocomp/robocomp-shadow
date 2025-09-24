# room_detector
Intro to component here


## Configuration parameters
As any other component, *room_detector* needs a configuration file to start. In
```
etc/config
```
you can find an example of a configuration file. We can find there the following lines:
```
EXAMPLE HERE
```

## Dependencies

To launch this component you will need to install the next packages:

```
pip install pyqtgraph open3d "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```
**If you have openCV installed normally, you need to uninstall and install the headless version**
```
pip uninstall opencv-python

pip install opencv-python-headless
```

## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

```
cd <room_detector's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/room_detector config
```
