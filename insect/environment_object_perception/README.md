# environment_object_perception
Intro to component here

You can download yolov8-seg.pt from https://github.com/ultralytics/ultralytics 

## Configuration parameters
As any other component, *yolov8_360* needs a configuration file to start. In
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
cd <yolov8_360's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/yolov8_360 config
```
