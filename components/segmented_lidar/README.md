# segmented_lidar
Intro to component here

This component requires instaling mmseg:

- If using python 3.12:
    - python3 -m ensurepip --upgrade
    - pip3 install --upgrade setuptools

- pip3 install cupy-cuda12x
- pip install ftfy
- pip3 install -U openmim
- mim install mmengine
- mim install mmcv==2.1.0
- mim install mmdet==3.2.0
- mim install mmsegmentation

## Configuration parameters
As any other component, *segmented_lidar* needs a configuration file to start. In
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
cd <segmented_lidar's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/segmented_lidar config
```
