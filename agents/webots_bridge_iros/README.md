# Webots2Robocomp
This project aims to build middleware that facilitates communication and integration between the Webots robotics simulator and the Robocomp framework.

## Webots Robot Configuration
For the middleware to effectively communicate with the robot in Webots, it's essential to configure the robot's controller to <extern>. This ensures that the robot's behavior is externally controlled by a program outside Webots, in this case, our middleware.

### Steps to Configure Robot in Webots
**Open your Webots world file:** This is typically a .wbt file where your robot and environment are defined.

**Select your robot node:** In the scene tree (usually on the left side of the Webots interface), find and select your robot node.

**Modify the controller field:** In the robot's properties (usually on the right side of the Webots interface), find the controller field. Change its value to <extern>.

**Save the world file:** Ensure you save the .wbt file after making this change.

Now, when you run the simulation in Webots, the robot will wait for an external controller (**Webots2Robocomp** in this case) to connect and take control.

## Connecting Middleware to Webots
Once the robot's controller is set to <extern>, ensure your middleware is ready to connect. Typically, you'd do this:

Start Webots and play the simulation.

Launch the middleware. It should automatically detect the robot waiting for an external controller and establish a connection.

## Starting the component
To avoid changing the *config* file in the repository, we can copy it to the component's home directory, so changes will remain untouched by future git pulls:

```
cd <Webots2Robocomp's path>
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/Webots2Robocomp config
```

## Configuration parameters
As any other component, *Webots2Robocomp* needs a configuration file to start. In
```
etc/config
```
you can find an example of a configuration file. We can find there the following lines:
```
CommonBehavior.Endpoints=tcp -p 10217

# Endpoints for implements interfaces
Laser.Endpoints=tcp -p 10003
Lidar3D.Endpoints=tcp -p 11988
CameraRGBDSimple.Endpoints=tcp -p 10096


InnerModelPath = innermodel.xml

Ice.Warn.Connections=0
Ice.Trace.Network=0
Ice.Trace.Protocol=0
Ice.MessageSizeMax=20004800
```

## Webots Scene Component Naming Conventions
For the middleware to properly interface with the components in Webots, it's crucial that certain components adhere to specific naming conventions. This ensures that the middleware can correctly identify and interact with these components.
Required Component Names:

<p>
<strong>Lidar:</strong> The Lidar component in your Webots scene must be named lidar.

<strong>Camera:</strong> The Camera component should be named camera.

<strong>Range Finder:</strong> This component needs to be named range-finder.
</p>

## Robocomp to Webots Component Mapping
For the middleware to function correctly, there's a specific mapping between the components implemented in Robocomp and their corresponding components in Webots. Here's a breakdown of this mapping:

| Robocomp Component | Webots Component |                                    Description |
|--------------------|:----------------:|-----------------------------------------------:|
| Laser              | Lidar            |        Corresponds to the 2D Lidar in Webots.  |
| Lidar3D            | 3D part of Lidar |    Represents the 3D functionalities of Lidar. |
| CameraRGBDSimple   |   Camera (RGB)   |                 Maps to the cameras in Webots. |
|                    | RangeFinder (D)  | The depth (D) specifically maps to RangeFinder |


### Note
It's essential to ensure that the Webots components adhere to the naming conventions mentioned in the previous section. This ensures seamless communication between Robocomp and Webots through the middleware.