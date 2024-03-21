import time
import Ice
import IceStorm
from rich.console import Console, Text
console = Console()


Ice.loadSlice("-I ./src/ --all ./src/JoystickAdapter.ice")
import RoboCompJoystickAdapter

class AxisList(list):
    def __init__(self, iterable=list()):
        super(AxisList, self).__init__(iterable)

    def append(self, item):
        """
        appends an instance of `RoboCompJoystickAdapter.AxisParams` to a list
        maintained by the object of the class `AxisList`.

        Args:
            item (`RoboCompJoystickAdapter.AxisParams`.): `RoboCompJoystickAdapter.AxisParams`
                instance to be appended to the list of axes managed by the `AxisList`
                class.
                
                		- `isinstance(item, RoboCompJoystickAdapter.AxisParams)`: This
                line checks whether the input `item` is an instance of
                `RoboCompJoystickAdapter.AxisParams`. If it is not, a TypeError
                will be raised.
                

        """
        assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).append(item)

    def extend(self, iterable):
        """
        modifies an instance of `AxisList` by adding the elements of an iterable
        sequence consisting of instances of `RoboCompJoystickAdapter.AxisParams`.

        Args:
            iterable (iterable.): Iterable containing the Axis Parameters to be
                extended for the `AxisList` instance.
                
                		- The function takes in an iterable of objects, which is asserted
                to be instances of `RoboCompJoystickAdapter.AxisParams`. This
                indicates that the input objects contain information about an axis
                (e.g., joystick axis) and its parameters (e.g., minimum/maximum values).
                		- The `super` call extends the parent class's `extend` function
                with the passed iterable, regardless of its composition or properties.
                

        """
        for item in iterable:
            assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).extend(iterable)

    def insert(self, index, item):
        """
        insert a new axis into an AxisList object at a specified index, providing
        the newly inserted axis with a RoboCompJoystickAdapter.AxisParams instance
        as argument.

        Args:
            index (int): 0-based index at which the new axis will be inserted into
                the list of axes maintained by the `AxisList` instance.
            item (`RoboCompJoystickAdapter.AxisParams`.): AXIS PARAMS object that
                will be inserted into the AXIS LIST maintained by the class `RoboCompJoystickAdapter.AxisList`.
                
                		- `isinstance(item, RoboCompJoystickAdapter.AxisParams)`: Verifies
                that the input is an instance of the `AxisParams` class provided
                by `RoboCompJoystickAdapter`.
                

        """
        assert isinstance(item, RoboCompJoystickAdapter.AxisParams)
        super(AxisList, self).insert(index, item)

setattr(RoboCompJoystickAdapter, "AxisList", AxisList)
class ButtonsList(list):
    def __init__(self, iterable=list()):
        super(ButtonsList, self).__init__(iterable)

    def append(self, item):
        """
        in the `ButtonsList` class allows you to add a new item to the list of
        buttons. The function takes a single argument of type
        `RoboCompJoystickAdapter.ButtonParams`, which is asserted to be an instance
        of the expected type. The function then calls the `super` method to append
        the new item to the list, following the inheritance hierarchy of the class.

        Args:
            item (`RoboCompJoystickAdapter.ButtonParams`.): `RoboCompJoystickAdapter.ButtonParams`
                object to be appended to the `ButtonsList`.
                
                		- `isinstance(item, RoboCompJoystickAdapter.ButtonParams)`:
                Verifies that the input is an instance of the
                `RoboCompJoystickAdapter.ButtonParams` class.
                		- `super(ButtonsList, self).append(item)`: Inserts the `item`
                into the list `self`.
                

        """
        assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).append(item)

    def extend(self, iterable):
        """
        takes an iterable object and recursively loops through its elements,
        checking each element for type equivalence with the `ButtonParams` class
        before calling superclass methods to incorporate it into a list instance.

        Args:
            iterable (list): 2D list of buttons that can be added to the `ButtonsList`.

        """
        for item in iterable:
            assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).extend(iterable)

    def insert(self, index, item):
        """
        in the `ButtonsList` class inserts an item into the list at the specified
        index, ensuring that it is a valid ButtonParams instance and utilizing the
        superclass's implementation of the insertion method.

        Args:
            index (int): 0-based index of the location where the new button will
                be inserted in the list managed by the `ButtonsList` object.
            item (`RoboCompJoystickAdapter.ButtonParams`.): ButtonParams object
                that is being added to the buttons list at the specified index.
                
                		- `isinstance(item, RoboCompJoystickAdapter.ButtonParams)` -
                checks if the input `item` is an instance of the `ButtonParams`
                class from the `RoboCompJoystickAdapter` module.
                

        """
        assert isinstance(item, RoboCompJoystickAdapter.ButtonParams)
        super(ButtonsList, self).insert(index, item)

setattr(RoboCompJoystickAdapter, "ButtonsList", ButtonsList)

import joystickadapterI



class Publishes:
    def __init__(self, ice_connector, topic_manager):
        """
        initializes a TopicManager class instance, setting its `ice_connector` and
        `topic_manager` attributes.

        Args:
            ice_connector (object reference, according to the given source code.):
                icedriver connection object, which is used to interact with the
                ICE (Inter-process Communication) connector and manage communication
                between processes.
                
                		- `self.ice_connector`: The `IceConnector` object that contains
                information about the connection to the remote system.
                
            topic_manager (object reference, according to the given source code.):
                topic manager, which manages the lifecycle of topics, providing a
                way to subscribe and unsubscribe from them.
                
                		- `topic_manager`: A `TopicManager` instance that manages the
                topics related to the IceConnector.
                

        """
        self.ice_connector = ice_connector
        self.mprx={}
        self.topic_manager = topic_manager


    def create_topic(self, topic_name, ice_proxy):
        # Create a proxy to publish a AprilBasedLocalization topic
        """
        creates an IceProxy instance and registers it with a given topic name,
        making it available for publication.

        Args:
            topic_name (str): name of the topic to be created or retrieved by the
                `create_topic()` function.
            ice_proxy (Ice::Proxies::UncheckedCast<pub>.): iced Python object that
                will be used to create a publisher proxy for the given topic name,
                allowing access to the published data through the `ice_oneway()`
                method.
                
                		- `uncheckedCast`: This method returns an ice proxy instance
                that can be used to publish messages to a topic.
                		- `pub`: The `pub` property accesses the publisher of the topic,
                which is a `Ice.InputStream` object.
                

        Returns:
            `IceProxy`, specifically an instance of the `IceOneWay` publisher
            type.: a published IcePublisher proxy for the specified topic.
            
            		- `pub`: The publisher of the `AprilBasedLocalization` topic, which
            is an Ice::OneWay instance.
            		- `proxy`: The Ice proxy instance that wraps the `pub` instance,
            providing a way to access it from outside the component.
            		- `mprx`: A dictionary that maps topic names to their corresponding
            proxies.
            

        """
        topic = False
        try:
            topic = self.topic_manager.retrieve(topic_name)
        except:
            pass
        while not topic:
            try:
                topic = self.topic_manager.retrieve(topic_name)
            except IceStorm.NoSuchTopic:
                try:
                    topic = self.topic_manager.create(topic_name)
                except:
                    print(f'Another client created the {topic_name} topic? ...')
        pub = topic.getPublisher().ice_oneway()
        proxy = ice_proxy.uncheckedCast(pub)
        self.mprx[topic_name] = proxy
        return proxy

    def get_proxies_map(self):
        return self.mprx


class Requires:
    def __init__(self, ice_connector):
        """
        initializes a Python object, assigning `self.ice_connector` and creating
        an empty dictionary for `self.mprx`.

        Args:
            ice_connector (`object` (or any derived class thereof).): icy connection
                that the object will use to communicate with its peers.
                
                	1/ `mprx`: A dictionary-like object that stores metadata about
                the deserialized data.
                

        """
        self.ice_connector = ice_connector
        self.mprx={}

    def get_proxies_map(self):
        return self.mprx

    def create_proxy(self, property_name, ice_proxy):
        # Remote object connection for
        """
        establishes an object connection between an IceSimple client and a remote
        object. It returns a tuple of (successful, proxy) if the connection is
        successful; otherwise, it fails with an error message or None as the second
        value in the tuple.

        Args:
            property_name (ice_object reference.): name of the property to retrieve
                from the remote object in the IceBox connection.
                
                	1/ `ice_connector`: It is an instance of the `Ice.ConnectionManager`
                class and represents the connection to the remote object (e.g., a
                camera).
                	2/ `getProperties()`: This method retrieves a list of properties
                defined in the remote object's interface (defined in the `@public`
                attribute).
                	3/ `getProperty()`: It retrieves a specific property from the
                list obtained by calling `getProperties()`. The property name is
                specified as an argument to this method.
                	4/ `stringToProxy()`: This method converts a string representation
                of a proxy object into an actual Python object instance. The string
                passed as input represents the Python class and its arguments,
                separated by colons (":").
                	5/ `uncheckedCast()`: It upgrades the returned base proxy to an
                actual instance of the desired subclass. In this case, it is the
                `ice_proxy` class that should be upgraded.
                	6/ `mprx`: This is a dictionary that stores properties of the
                remote object. The property names are the keys, and they map to
                corresponding proxy instances in the value slot.
                	7/ `traceback.print_exc()`: It is a utility function for printing
                the stack trace of an exception, including its position and detailed
                error message. This line is left unchanged as part of the challenge
                instructions.
                	8/ `console.log()`: It logs a message to the console with the
                specified string as an argument.
                
            ice_proxy (unchecked castable to a Proxy instance.): iced connection
                object to which the remote object reference is to be converted.
                
                		- `uncheckedCast`: This attribute is used to convert a proxy
                into a reference to an object or class that cannot be null. It is
                often utilized in conjunction with casting methods like
                `ice_connector.stringToProxy()`. (Note: No mention of the question
                or any personal pronouns, just direct answers.)
                		- `basePrx`: This attribute holds the base proxy for a given
                object reference, which can be further cast to a specific class
                or object reference using the `uncheckedCast()` method.
                

        Returns:
            `(True, proxy)`, where `proxy` is of type `AnyRef`, specifically a
            subclass of `Object`.: a tuple containing either True or False and the
            created proxy object for the specified property name.
            
            		- The first element of the tuple is `True`, indicating whether the
            proxy was created successfully (`True`) or not (`False`).
            		- The second element of the tuple is the `Proxy` object, which
            represents the remote object connection for the specified property.
            This object can be used to call methods on the remote object and access
            its properties.
            

        """
        try:
            proxy_string = self.ice_connector.getProperties().getProperty(property_name)
            try:
                base_prx = self.ice_connector.stringToProxy(proxy_string)
                proxy = ice_proxy.uncheckedCast(base_prx)
                self.mprx[property_name] = proxy
                return True, proxy
            except Ice.Exception:
                print('Cannot connect to the remote object (CameraSimple)', proxy_string)
                # traceback.print_exc()
                return False, None
        except Ice.Exception as e:
            console.print_exception(e)
            console.log(f'Cannot get {property_name} property.')
            return False, None


class Subscribes:
    def __init__(self, ice_connector, topic_manager, default_handler):
        """
        initializes an instance of a `JoystickAdapter`, creating an adapter for
        the "JoystickAdapterTopic" and setting its `JoystickAdapterI` to a new
        instance of `JoystickAdapterI` with a default handler.

        Args:
            ice_connector (`iceconnector` object reference.): IceConnector, which
                is an interface for connecting to other ice objects and services
                in the system.
                
                		- `ice_connector`: A standard Python `ICEConnector` object that
                defines how the client will communicate with its ice router over
                an interprocess communication (IPC) channel.
                
            topic_manager (`TopicManager` object.): TopicManager instance that
                controls the management of topics related to the application's
                state, event handling, and communication with other components or
                services.
                
                		- `topic_manager`: A topic manager object that represents a set
                of topics and their associated messages. It is used to manage the
                communication between the topic adapter and the topics it subscribes
                to.
                
                	Note: The term `JoystickAdapterTopic` does not appear in the code
                snippet provided, so its meaning cannot be inferred.
                
            default_handler (`joystickadapterI.JoystickAdapterI`.): default handler
                that will be used for topics that are not handled by the JoystickAdapter
                Topic.
                
                		- `default_handler`: A reference to an instance of
                `joystickadapterI.JoystickAdapterI`, which is a class provided in
                the input. This instance serves as the default handler for topics
                related to joysticks.
                

        """
        self.ice_connector = ice_connector
        self.topic_manager = topic_manager

        self.JoystickAdapter = self.create_adapter("JoystickAdapterTopic", joystickadapterI.JoystickAdapterI(default_handler))

    def create_adapter(self, property_name, interface_handler):
        """
        creates an object adapter and adds a proxy to it. It also creates a topic
        if it does not exist, subscribes to the topic using a specified QoS, and
        activates the adapter.

        Args:
            property_name (str): name of the property that will be created as an
                ice topic.
            interface_handler (` Ice.ObjectAdapter `.): handler object that
                implements the required interfaces for the created adapter.
                
                		- `ice_connector`: An instance of `Ice.Connection` representing
                the connection to the Ice stack.
                		- `property_name`: The name of the property being adapted.
                		- `interface_handler`: An instance of an object implementing the
                `Ice.Object` interface, which contains methods and properties
                related to the adapter.
                		- `topic_name`: The name of the topic to be created or retrieved.
                		- `subscribe_done`: A boolean variable indicating whether the
                topic has been successfully subscribed to. It is initialized to
                `False` and is set to `True` after successful subscription or
                failed attempt.
                		- `qos`: A dictionary object containing QoS (Quality of Service)
                parameters for the publisher, which are set upon subscription.
                		- `proxy`: An instance of an Ice.Proxy object representing the
                proxy created for the adapter.
                

        Returns:
            int: an object adapter instance that can be used to subscribe to a topic.

        """
        adapter = self.ice_connector.createObjectAdapter(property_name)
        handler = interface_handler
        proxy = adapter.addWithUUID(handler).ice_oneway()
        topic_name = property_name.replace('Topic','')
        subscribe_done = False
        while not subscribe_done:
            try:
                topic = self.topic_manager.retrieve(topic_name)
                subscribe_done = True
            except Ice.Exception as e:
                console.log("Error. Topic does not exist (creating)", style="blue")
                time.sleep(1)
                try:
                    topic = self.topic_manager.create(topic_name)
                    subscribe_done = True
                except:
                    console.log(f"Error. Topic {Text(topic_name, style='red')} could not be created. Exiting")
                    status = 0
        qos = {}
        topic.subscribeAndGetPublisher(qos, proxy)
        adapter.activate()
        return adapter


class Implements:
    def __init__(self, ice_connector, default_handler):
        self.ice_connector = ice_connector

    def create_adapter(self, property_name, interface_handler):
        """
        creates an object adapter based on a given property name and interface
        handler, adding the interface handler to the adapter and activating it.

        Args:
            property_name (str): name of the property to create an adapter for,
                which is used as the identifier for the new adapter.
            interface_handler (`identity`.): handler for the interface associated
                with the property name, which is used to activate the adapter.
                
                		- `self.ice_connector`: A reference to an instance of
                `IntermediateController', which serves as a bridge between the
                application code and the IDL-defined interfaces.
                		- `property_name': The name of the property being created as an
                adapter.
                		- `interface_handler': A reference to the interface object that
                defines the interface for which an adapter is being created.
                

        """
        adapter = self.ice_connector.createObjectAdapter(property_name)
        adapter.add(interface_handler, self.ice_connector.stringToIdentity(property_name.lower()))
        adapter.activate()


class InterfaceManager:
    def __init__(self, ice_config_file):
        # TODO: Make ice connector singleton
        """
        initializes an Ice connector singleton by loading configuration from an
        Ice config file, and sets up topic manager for property storage, and
        Requires, Publishes, Implements and Subscribes objects to handle interactions
        with the Ice framework.

        Args:
            ice_config_file (str): file containing configuration settings for the
                ICE connector, which are used to initialize and set up the connector
                instance.

        """
        self.ice_config_file = ice_config_file
        self.ice_connector = Ice.initialize(self.ice_config_file)
        needs_rcnode = True
        self.topic_manager = self.init_topic_manager() if needs_rcnode else None

        self.status = 0
        self.parameters = {}
        for i in self.ice_connector.getProperties():
            self.parameters[str(i)] = str(self.ice_connector.getProperties().getProperty(i))
        self.requires = Requires(self.ice_connector)
        self.publishes = Publishes(self.ice_connector, self.topic_manager)
        self.implements = None
        self.subscribes = None



    def init_topic_manager(self):
        # Topic Manager
        """
        initializes a Topic Manager object by converting a string representation
        of a Proxy object into an instance of the TopicManager class using IceStorm's
        `checkedCast`. If the connection to the RC node is refused, it logs an
        error message and exits with a negative status code.

        Returns:
            IceStorm TopicManagerPrx object.: an IceStorm.TopicManagerPrx object.
            
            		- `TopicManagerPrx`: This is the IceStorm Topic Manager interface
            pointer that is returned by the function after successful initialization.
            It represents a proxy object that provides access to the topic manager's
            functionality through the IceStorm framework.
            		- `obj`: This variable stores the input parameter passed to the
            `ice_connector.stringToProxy()` method, which is used to create the
            Topic Manager proxy object.
            

        """
        proxy = self.ice_connector.getProperties().getProperty("TopicManager.Proxy")
        obj = self.ice_connector.stringToProxy(proxy)
        try:
            return IceStorm.TopicManagerPrx.checkedCast(obj)
        except Ice.ConnectionRefusedException as e:
            console.log(Text('Cannot connect to rcnode! This must be running to use pub/sub.', 'red'))
            exit(-1)

    def set_default_hanlder(self, handler):
        """
        sets the default event handler for an instance of a class by storing it
        in the `implements` and `subscribes` attributes.

        Args:
            handler (object.): handle that is used to interact with the remote
                object and execute its methods.
                
                	1/ `implements`: An instance of `Implements`, which is a container
                for the ice connector and handler objects.
                	2/ `subscribes`: An instance of `Subscribes`, which manages the
                subscriptions to the topic manager and handler objects.
                

        """
        self.implements = Implements(self.ice_connector, handler)
        self.subscribes = Subscribes(self.ice_connector, self.topic_manager, handler)

    def get_proxies_map(self):
        """
        updates a dictionary with proxies information based on the requirements
        and publications of the class instance, and returns the updated map.

        Returns:
            dict: a dictionary containing the proxies map for both required and
            published interfaces.

        """
        result = {}
        result.update(self.requires.get_proxies_map())
        result.update(self.publishes.get_proxies_map())
        return result

    def destroy(self):
        """
        terminates the IceConnector instance, effectively decommissioning it from
        service.

        """
        if self.ice_connector:
            self.ice_connector.destroy()




