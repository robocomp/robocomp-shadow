# yolov8_360_seg
Intro to component here

To install the software for getting orientation: 
- Clone https://github.com/hnuzhy/JointBDOE in /home/robocomp/software
- Download the weights from https://huggingface.co/HoyerChou/JointBDOE 
- Copy them to /home/robocomp/software/JointBDOE/runs/JointBDOE/coco_s_1024_e500_t020_w005/weights
- Rename the chosen one to "best.pt"

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
