# environment_object_perception
Intro to component here

## Dependencies

```bash
#Clone repo
git clone https://github.com/hnuzhy/JointBDOE.git ~/software/JointBDOE
#
wget https://huggingface.co/HoyerChou/JointBDOE/resolve/main/coco_m_1024_e500_t010_w005_best.pt -O $HOME/software/JointBDOE/runs/JointBDOE/coco_s_1024_e500_t020_w005/weights/best.pt
```

You can download yolov8-seg.pt from https://github.com/ultralytics/ultralytics 

## Configuration parameters
As any other component, *environment_object_perception* needs a configuration file to start. In
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
cd <environment_object_perception's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/environment_object_perception config
```
