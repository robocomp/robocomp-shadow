# giraff_pyrep

This is a pseudo-component, a Python adapter, that used the PyRep module to start CoppeliaSim and access the simulated scene. This method is much faster that the remote communcation through middlewares.
The main limitation is that it cannot be regenerated using RoboComp's code generator, because it is Qt-free. The reason is the incompatibility between the CoppeliaSim own version of Qt and the version usually installed in the machine.

Please follow these steps:

- Install https://github.com/stepjam/PyRep
- You need to copy the shadow.py file in the component's root folder to:

   /home/xxxyour-userxxx/.local/lib/python3.10/site-packages/pyrep/robots/mobiles/shadow.py
   (for a local install of Pyrep in python3.10)
 
- Comment out the following lines in /home/xxxyour-userxxx/.local/lib/python3.6/site-packages/pyrep/objects/object.py :
  assert_type = self._get_requested_type()
  actual = ObjectType(sim.simGetObjectType(self._handle))
  if actual != assert_type:
      raise WrongObjectTypeError(
          'You requested object of type %s, but the actual type was '
          '%s' % (assert_type.name, actual.name))

- Comment out line 41 in file ~/.local/lib/python3.10/site-packages/pyrep/robots/mobiles/mobile_base.py

You can find more detailed instructions in 
     https://github.com/robocomp/robocomp/blob/development/doc/robocomp-pyrep.md

- To start the component copy the config file form etc to . and use the script run.sh. 

- Make sure first that you have commented this two lines, in case you have them:
 
  export LD_LIBRARY_PATH=$COPPELIASIM_ROOT
  export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

. If you have a joystick, start ~/robocomp/components/robocomp-robolab/components/hardware/external_control/joystickpublish
. Check the config file to set the ranges of the axis.


## Configuration parameters
As any other component, *viriatoPyrep* needs a configuration file to start. In
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
cd <viriatoPyrep's path> 
```
```
cp etc/config config
```

After editing the new config file we can run the component:

```
bin/viriatoPyrep config
```

## Adding new publish interfaces without "robocompdsl"

- In main file "my_component.py" Add a block like this:

```
 # Create a proxy to publish a KinovaArmPub topic
    topic = False
    try:
        topic = topicManager.retrieve("KinovaArmPub")
    except:
        pass
    while not topic:
        try:
            topic = topicManager.retrieve("KinovaArmPub")
        except IceStorm.NoSuchTopic:
            try:
                topic = topicManager.create("KinovaArmPub")
            except:
                print('Another client created the JointMotorPub topic? ...')
    pub = topic.getPublisher().ice_oneway()
    kinovaarmpubTopic = RoboCompKinovaArmPub.KinovaArmPubPrx.uncheckedCast(pub)
    mprx["KinovaArmPubPub"] = kinovaarmpubTopic
```

## Adding new implemented interfaces without "robocompdsl"

- In main file "my_component.py" Add a block like this:

```
    adapter = ic.createObjectAdapter('CameraSimple')
    adapter.add(camerasimpleI.CameraSimpleI(worker), ic.stringToIdentity('camerasimple'))
    adapter.activate()
```
- 
- In genericworker.py add 

```
Ice.loadSlice("-I ./src/ --all ./src/KinovaArmPub.ice")
import RoboCompKinovaArmPub
```
and in the constructor

```
  self.kinovaarmpub_proxy = mprx["KinovaArmPubPub"]
```

- Add to CMakeLists.txt the name of the interface
- run cmake .
