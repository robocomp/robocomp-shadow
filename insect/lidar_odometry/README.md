# lidar_odometry
Intro to component here

Computes the robot's odometry from the laser scans using the fast_gicp library:  https://github.com/SMRT-AIST/fast_gicp

## Configuration parameters
As any other component, *lidar_odometry* needs a configuration file to start. In
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
cd <lidar_odometry's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/lidar_odometry config
```
