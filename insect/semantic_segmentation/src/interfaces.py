import time
import Ice
import IceStorm
from rich.console import Console, Text
console = Console()


Ice.loadSlice("-I ./src/ --all ./src/Camera360RGB.ice")
import RoboCompCamera360RGB
Ice.loadSlice("-I ./src/ --all ./src/Camera360RGBD.ice")
import RoboCompCamera360RGBD
Ice.loadSlice("-I ./src/ --all ./src/MaskElements.ice")
import RoboCompMaskElements
Ice.loadSlice("-I ./src/ --all ./src/Person.ice")
import RoboCompPerson
Ice.loadSlice("-I ./src/ --all ./src/VisualElements.ice")
import RoboCompVisualElements

class ImgType(list):
    def __init__(self, iterable=list()):
        super(ImgType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(ImgType, self).append(item)

    def extend(self, iterable):
        """
        Checks whether each element in an iterable is an instance of a byte type,
        and if so, adds it to the image type object.

        Args:
            iterable (`byte`.): 1D array or list of bytes that are to be processed
                and added to the existing image data.
                
                	1/ The function receives an iterable object that contains instances
                of the `byte` type as elements.
                	2/ Each element in the iterable is verified to be an instance of
                the `byte` type using the `isinstance` method.
                	3/ The `super` method calls the parent class's `extend` method
                to append the elements in the iterable to the list of properties
                or attributes.

        """
        for item in iterable:
            assert isinstance(item, byte)
        super(ImgType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(ImgType, self).insert(index, item)

setattr(RoboCompCamera360RGB, "ImgType", ImgType)
class ImgType(list):
    def __init__(self, iterable=list()):
        super(ImgType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(ImgType, self).append(item)

    def extend(self, iterable):
        """
        Performs no operation and simply returns a reference to the receiver object,
        given that it takes an iterable containing instances of `byte`.

        Args:
            iterable (`bytes` or `byte`.): iterable of bytes that are asserted to
                be instances of the `byte` class before being extended by the
                `super()` method.
                
                		- For each item in `iterable`, `isinstance(item, byte)` asserts
                that each item is an instance of the `byte` type. This ensures
                that only bytes can be added to the image object.
                
                	Overall, the `extend` function takes an iterable of bytes and
                adds them to the image object, verifying that each item in the
                iterable is a byte.

        """
        for item in iterable:
            assert isinstance(item, byte)
        super(ImgType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(ImgType, self).insert(index, item)

setattr(RoboCompCamera360RGBD, "ImgType", ImgType)
class ImgType(list):
    def __init__(self, iterable=list()):
        super(ImgType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(ImgType, self).append(item)

    def extend(self, iterable):
        """
        Adds a new instance of type `ImgType` to an existing sequence. It checks
        each item in the iterable is of type `byte`.

        Args:
            iterable (`byte`.): list or sequence of bytes to be processed by the
                `super()` method, thereby extending the `ImgType` instance with
                the provided byte objects.
                
                	The type of each element in `iterable` is verified as `byte`.

        """
        for item in iterable:
            assert isinstance(item, byte)
        super(ImgType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(ImgType, self).insert(index, item)

setattr(RoboCompMaskElements, "ImgType", ImgType)
class TMasks(list):
    def __init__(self, iterable=list()):
        super(TMasks, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompMaskElements.TMask)
        super(TMasks, self).append(item)

    def extend(self, iterable):
        """
        Takes an iterable of Robocomp Mask Elements (TMasks) and adds them to the
        class instance.

        Args:
            iterable (list): list of `RoboCompMaskElements.TMask` objects that
                will be added to the TMasks instance through its extend() method.

        """
        for item in iterable:
            assert isinstance(item, RoboCompMaskElements.TMask)
        super(TMasks, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompMaskElements.TMask)
        super(TMasks, self).insert(index, item)

setattr(RoboCompMaskElements, "TMasks", TMasks)
class TMaskNames(list):
    def __init__(self, iterable=list()):
        super(TMaskNames, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, str)
        super(TMaskNames, self).append(item)

    def extend(self, iterable):
        """
        Is used to add items to an existing sequence (such as a list) and returns
        the updated sequence without creating a new one.

        Args:
            iterable (str): iterable of str items to be extended by the function.

        """
        for item in iterable:
            assert isinstance(item, str)
        super(TMaskNames, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, str)
        super(TMaskNames, self).insert(index, item)

setattr(RoboCompMaskElements, "TMaskNames", TMaskNames)
class TConnections(list):
    def __init__(self, iterable=list()):
        super(TConnections, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompPerson.TConnection)
        super(TConnections, self).append(item)

    def extend(self, iterable):
        """
        Updates a list with new items, and checks each item's type to be instance
        of `RoboCompPerson.TConnection`.

        Args:
            iterable (list): list of `RoboCompPerson.TConnection` objects that are
                added to the instance of the `TConnections` class through the
                `extend()` method.

        """
        for item in iterable:
            assert isinstance(item, RoboCompPerson.TConnection)
        super(TConnections, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompPerson.TConnection)
        super(TConnections, self).insert(index, item)

setattr(RoboCompPerson, "TConnections", TConnections)
class TMetrics(list):
    def __init__(self, iterable=list()):
        super(TMetrics, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, float)
        super(TMetrics, self).append(item)

    def extend(self, iterable):
        """
        Applies a sequence of floats to the model's `TMetrics` object, verifying
        that each element is an instance of `float`.

        Args:
            iterable (float): sequence of metrics that will be extended by the
                function, each metric being an instance of `float`.

        """
        for item in iterable:
            assert isinstance(item, float)
        super(TMetrics, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, float)
        super(TMetrics, self).insert(index, item)

setattr(RoboCompVisualElements, "TMetrics", TMetrics)
class TObjectList(list):
    def __init__(self, iterable=list()):
        super(TObjectList, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompVisualElements.TObject)
        super(TObjectList, self).append(item)

    def extend(self, iterable):
        """
        Updates the internal list of elements in the class by appending new iterable
        objects of type `RoboCompVisualElements.TObject`.

        Args:
            iterable (list): collection of objects to be added to the TObjectList.

        """
        for item in iterable:
            assert isinstance(item, RoboCompVisualElements.TObject)
        super(TObjectList, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompVisualElements.TObject)
        super(TObjectList, self).insert(index, item)

setattr(RoboCompVisualElements, "TObjectList", TObjectList)

import maskelementsI
import visualelementsI



class Publishes:
    def __init__(self, ice_connector, topic_manager):
        """
        Sets instance attributes `ice_connector`, `mprx`, and `topic_manager` based
        on passed-in arguments, initializing a PyRSS21 connector for ice-util
        library, creating an empty mapping for mprx, and setting the topic manager
        to a specified value.

        Args:
            ice_connector (`object`.): icy connector for the current topic manager,
                which is used to manage the connection between the C++ code and
                the ROS framework.
                
                		- `ice_connector`: A class attribute that holds an instance of
                `ICEConnector`.
            topic_manager (`object`.): management of topics that are to be managed
                by the instance of the class.
                
                		- `mprx`: A dictionary containing information about the managed
                property refresh times for each topic.
                		- `ice_connector`: An instance of `ICEConnector`.

        """
        self.ice_connector = ice_connector
        self.mprx={}
        self.topic_manager = topic_manager


    def create_topic(self, topic_name, ice_proxy):
        # Create a proxy to publish a AprilBasedLocalization topic
        """
        Creates a new topic and publisher or retrieves an existing one based on
        its name, using Ice Storm's topic management facility.

        Args:
            topic_name ("ice_name" or more specifically an instance of "
                IceStorm.NoSuchTopic".): name of a topic to be created or retrieved
                in the function, and is used to identify the topic in the code's
                execution.
                
                		- `topic_name`: This is the name of the topic to be created. It
                is a string attribute that identifies the topic.
                
                	Explanation of the various attributes/properties of `topic_name`:
                
                		- `String`: The topic name is represented as a string value.
                		- `Unique within the application`: Each topic name must be unique
                within the application to avoid conflicts.
                		- `Defined by the client or server`: Either the client or server
                can define the topic name, depending on the Ice framework implementation.
                
                	Note: This explanation does not provide a summary at the end as
                requested in the given question format.
            ice_proxy (`ICE_PROXY`.): iced-client publisher and converts it to an
                unchecked IcePy cast to create a proxy for the topic publisher.
                
                		- `uncheckedCast`: This method returns an Ice::Proxy object,
                which is a reference to the object on another endpoint that is
                created using the `ice_oneway()` method.
                		- `pub`: This property accesses the publisher associated with
                the topic, allowing you to call methods such as `ice_send()` on it.
                		- `topic_name`: This is the name of the topic being created or
                referenced.

        Returns:
            IceProxy object: an Ice Proxy object that represents a published topic.
            
            		- `pub`: The publisher instance of the topic. It is an `IceStorm.Publisher`
            object.
            		- `proxy`: The ice proxy instance of the publisher. It is an
            `ice_proxy.UncheckedCast` object.
            		- `mprx`: A dictionary containing the created publisher Proxy. The
            key is the name of the topic, and the value is the Proxy object.

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
        Sets up instances of proxies for the `Camera360RGBDProxy` and
        `VisualElementsProxy`, allowing the object to interact with them remotely
        through their corresponding prx objects.

        Args:
            ice_connector (`ice_connector`.): iced connector that enables the
                function to generate high-quality documentation for the given code.
                
                		- `ice_connector`: An instance of `IceConnector`. Its properties
                and attributes include `name`, `host`, `port`, `username`, `password`,
                `root_dir`, `current_dir`, `max_threads`, `read_only`, `compress`,
                `server`, and others.

        """
        self.ice_connector = ice_connector
        self.mprx={}

        self.Camera360RGBD = self.create_proxy("Camera360RGBDProxy", RoboCompCamera360RGBD.Camera360RGBDPrx)

        self.VisualElements = self.create_proxy("VisualElementsProxy", RoboCompVisualElements.VisualElementsPrx)

    def get_proxies_map(self):
        return self.mprx

    def create_proxy(self, property_name, ice_proxy):
        # Remote object connection for
        """
        1) retrieves a property name from the `ice_connector` object, 2) uses the
        retrieved property name to create a proxy using the `stringToProxy()`
        method, and 3) stores the created proxy in the `mprx` dictionary for later
        use.

        Args:
            property_name (str): name of the property to be retrieved from the
                remote object.
            ice_proxy (IceProxy object.): icy proxy object that is returned by
                `self.ice_connector.stringToProxy()` when converting a string
                representation of a proxy into an actual IcePy proxy object.
                
                		- `uncheckedCast`: This attribute indicates that the proxy can
                be created without checking if it is a valid instance of the correct
                class.
                		- `base_prx`: This attribute holds the base Proxy object that
                was obtained by calling `self.ice_connector.stringToProxy()` on
                the input string.

        Returns:
            ICE proxy: a tuple of two values: (successful creation, the created
            proxy object).
            
            		- `True, proxy`: Indicates whether the creation of the proxy was
            successful (`True`) and the obtained proxy object (`proxy`).
            		- `False, None`: Indicates that an error occurred during the creation
            of the proxy.
            		- `property_name`: The name of the property being requested.
            		- `ice_connector`: An instance of the `Ice` connector class, used
            to create the proxy.

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
        self.ice_connector = ice_connector
        self.topic_manager = topic_manager

    def create_adapter(self, property_name, interface_handler):
        """
        Creates an Iceoryx adapter object based on a property name provided by the
        user and returns it after activation.

        Args:
            property_name (str): name of a property to be used for topic naming
                and is used to create a unique topic name for subscribing to a topic.
            interface_handler (Ice.Handler instance.): handler for an interface
                that is to be handled by the adapter created by the function.
                
                		- `handlers`: A list of handler objects that define the interfaces
                implemented by the adapter.
                		- `oneway`: An Ice.OneWay implementation object that provides
                one-way communication between the client and server.
                		- `topic_name`: The name of the topic to which the adapter subscribes.
                		- `subscribe_done`: A boolean variable indicating whether the
                adapter has successfully subscribed to the topic or not.
                		- `qos`: An Ice.QoS implementation object that defines the quality
                of service (QoS) requirements for the subscription.
                
                	The `adapter` object created by the function is returned at the
                end, which represents an active adapter with a subscription to the
                specified topic using the `oneway` and `qos` properties.

        Returns:
            IceAdapter: an instance of the ` ice.adapter ` class that represents
            a communication channel between a client and a server.
            
            		- `adapter`: This is an Ice::Adapter instance, which represents a
            connection point between the client and server.
            		- `handler`: This is an Ice::Object::InterfaceHandler instance, which
            manages the communication between the client and server for a specific
            interface.
            		- `proxy`: This is an Ice::Object::Proxy instance, which represents
            the interface implementation on the server side.
            		- `topic_name`: This is the name of the topic that the adapter listens
            to, obtained by removing the "Topic" prefix from the input property name.
            		- `subscribe_done`: This is a boolean flag indicating whether the
            topic already exists or not.
            		- `qos`: This is an instance of Ice::QoS instance, which represents
            the quality of service (QoS) parameters for the topic subscription.
            		- `adapter.activate()`: This method activates the adapter, enabling
            communication between the client and server.

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
        """
        Initializes the object's instance variables, `ice_connector`, `maskelements`,
        and `visualelements`, which are adapter classes for interacting with the
        `MaskElements` and `VisualElements` modules.

        Args:
            ice_connector (`object`.): 3A Mask and Visual Elements Connector, which
                allows the creation of adapters for MaskElements and VisualElements.
                
                		- `ice_connector`: This is an instance of the `ice_connector`
                class, which contains various properties and attributes related
                to ice connector.
            default_handler (`ice_tracing.Handler` object.): default handler for
                MaskElements and VisualElements instances created by the
                `create_adapter()` method.
                
                		- `ice_connector`: This attribute refers to an instance of the
                `ice_connector` class.

        """
        self.ice_connector = ice_connector
        self.maskelements = self.create_adapter("MaskElements", maskelementsI.MaskElementsI(default_handler))
        self.visualelements = self.create_adapter("VisualElements", visualelementsI.VisualElementsI(default_handler))

    def create_adapter(self, property_name, interface_handler):
        """
        Creates an adaptor and adds an interface handler to it, and activates it.

        Args:
            property_name (str): property name to which an adapter is being created
                for.
            interface_handler (identity.): identity of an interface that is being
                added to the adapter.
                
                		- `self.ice_connector`: The Ice connector object that is being
                utilized to create an adapter.
                		- `property_name`: A string specifying the name of the property
                to which the adapter is being added.
                		- `stringToIdentity()`: A method that converts a given string
                into an identity value, which can be used to identify the adapter
                within the Ice environment.

        """
        adapter = self.ice_connector.createObjectAdapter(property_name)
        adapter.add(interface_handler, self.ice_connector.stringToIdentity(property_name.lower()))
        adapter.activate()


class InterfaceManager:
    def __init__(self, ice_config_file):
        # TODO: Make ice connector singleton
        """
        Sets up the necessary objects and initializes properties based on an ice
        configuration file provided as input. It creates a `TopicManager`, `Requires`,
        `Publishes`, `Implements`, and `Subscribes` instances, all of which are
        connected to the `IceConnector`.

        Args:
            ice_config_file (str): file that contains configuration details for
                Ice, which is then used to initialize and set up the Ice framework
                within the object.

        """
        self.ice_config_file = ice_config_file
        self.ice_connector = Ice.initialize(self.ice_config_file)
        needs_rcnode = False
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
        Parses a given Proxy string, then tries to cast it to an IceStorm.TopicManagerPrx
        object if successful, otherwise it logs an error message and exits with a
        negative value.

        Returns:
            IceStorm.TopicManagerPrx: an initialized `IceStorm.TopicManagerPrx` object.
            
            		- `TopicManagerPrx`: This is an interface proxy that represents the
            TopicManager object. It can be used to call methods on the TopicManager
            object remotely.
            		- `IceStorm`: This is the module that provides the API for creating
            and managing ice topics.
            		- `ConnectionRefusedException`: This is an exception that is thrown
            if there is a problem connecting to the rcnode server. It is caught
            and logged by the `init_topic_manager` function.

        """
        proxy = self.ice_connector.getProperties().getProperty("TopicManager.Proxy")
        obj = self.ice_connector.stringToProxy(proxy)
        try:
            return IceStorm.TopicManagerPrx.checkedCast(obj)
        except Ice.ConnectionRefusedException as e:
            console.log(Text('Cannot connect to rcnode! This must be running to use pub/sub.', 'red'))
            exit(-1)

    def set_default_hanlder(self, handler):
        self.implements = Implements(self.ice_connector, handler)
        self.subscribes = Subscribes(self.ice_connector, self.topic_manager, handler)

    def get_proxies_map(self):
        """
        Updates an empty dictionary `result` with two maps obtained from the class's
        attributes: `requires.get_proxies_map()` and `publishes.get_proxies_map()`.
        The resulting dictionary contains the union of the maps from both sources.

        Returns:
            UpdatePrimitive` object: a dictionary containing key-value pairs
            representing proxies for both requires and publishes.
            
            		- `result`: The dictionary containing the proxies map.
            		- `update()` methods: These methods update the properties of the
            result dictionary with new values obtained from other functions.
            		- `requires.get_proxies_map()` and `publishes.get_proxies_map()`:
            These are the functions called to generate the proxies map.

        """
        result = {}
        result.update(self.requires.get_proxies_map())
        result.update(self.publishes.get_proxies_map())
        return result

    def destroy(self):
        if self.ice_connector:
            self.ice_connector.destroy()




