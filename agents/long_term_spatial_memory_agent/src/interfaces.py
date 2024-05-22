import time
import Ice
import IceStorm
from rich.console import Console, Text
console = Console()







class Publishes:
    def __init__(self, ice_connector, topic_manager):
        """
        Sets up the instance variables `ice_connector`, `mprx`, and `topic_manager`.

        Args:
            ice_connector (`ICEConnector`.): ices connection to be established and
                is used by the `TopicManager` to determine which topics are relevant
                for the ice connection.
                
                		- `ice_connector`: This attribute holds the instance of an
                IceConnector class that provides connections to external tools and
                platforms using Ice Framework.
            topic_manager (object.): Topic Manager object that manages the creation,
                deletion, and update of topics in the system.
                
                		- `mprx`: This is an instance of `MessagePublisherReceiver`,
                which manages message publications and subscriptions for the Ice
                Connector.
                		- `ice_connector`: An instance of `IceConnector`, responsible
                for connecting to the remote system using Inter-Container Exchange
                (ICE).

        """
        self.ice_connector = ice_connector
        self.mprx={}
        self.topic_manager = topic_manager


    def create_topic(self, topic_name, ice_proxy):
        # Create a proxy to publish a AprilBasedLocalization topic
        """
        Retrieves or creates a topic based on its name, and returns the publisher
        of that topic as an ice::oneway proxy.

        Args:
            topic_name (str): name of a topic for which a publisher is being
                retrieved or created.
            ice_proxy (iced-proxy.): iced-python publisher as an IcePy proxy,
                allowing it to be cast into a different type of object without
                checking for ice-specific errors.
                
                		- `ice_oneway()`: This method returns an `ice::OneWay` object,
                which is a type of interface that provides a single-shot invocation
                mechanism.
                		- `uncheckedCast(pub)`: This method casts the `ice::Publisher`
                object to an `ice_proxy` object, allowing for later access to its
                properties and methods.

        Returns:
            IceProxy` object, specifically an instance of the `uncheckedCast`
            subclass of the `ice_proxy.IceProxy` class: a reference to an
            IceProxies.uncheckedCast instance representing the created topic's publisher.
            
            		- `pub`: This is the publisher for the created topic, which can be
            used to send messages to the topic.
            		- `proxy`: This is an ice proxy that wraps the `pub` object, allowing
            it to be accessed from outside the client.
            		- `self.mprx`: This is a dictionary that contains mappings of topic
            names to their corresponding publishers. The created topic is added
            to this dictionary.

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
        self.ice_connector = ice_connector
        self.mprx={}

    def get_proxies_map(self):
        return self.mprx

    def create_proxy(self, property_name, ice_proxy):
        # Remote object connection for
        """
        Creates a Proxy object for a remote object using the given Ice string
        representation. If successful, it returns a tuple of `True` and the Proxy
        object, otherwise it returns `False` with an error message.

        Args:
            property_name (str): name of the property to be retrieved from the
                remote object.
            ice_proxy (`Ice.Proxies.Unchecked`.): Ice.Reference that will be
                returned as the output of the function if the object is reachable
                through the given proxy string, and it will be used to uncheck the
                cast to an `Ice.ObjectPrx`.
                
                		- `uncheckedCast`: This is an attribute that returns the proxy
                instance created from the `base_prx`.
                		- `mprx`: This is a dictionary that contains the property names
                as keys and their corresponding proxies as values.
                
                	Inside the `try`-`except` block, the `ice_connector` property
                `getProperties()` method is called to retrieve the value of the
                specified property name (`property_name`). The resulting `proxy_string`
                is then converted to a proxy instance using the `stringToProxy()`
                method. If the conversion fails due to an error, the function
                returns `False`, `None`.
                
                	The rest of the code handles the exceptions raised by the
                `ice_connector` and `uncheckedCast` methods, respectively.

        Returns:
            ice.Proxy: a tuple containing a Boolean value indicating success and
            the proxy object for the specified property.
            
            		- `True, proxy`: If the function is successful in creating a proxy
            for the remote object, it returns `True` and the proxy itself as the
            second item in the tuple. The proxy is an instance of the
            `ice_proxy.uncheckedCast` class and has the property name as its attribute.
            		- `False, None`: If there is an error connecting to the remote object
            or retrieving its properties, the function returns `False` and `None`
            as the second item in the tuple.

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
        Creates a new ICE (Interactive Connectivity Engine) adapter based on a
        given property name, and then retrieves a topic by that name from the topic
        manager. It then subscriges to the topic using the ICE QoS (Quality of
        Service) settings, and activates the adapter.

        Args:
            property_name (str): name of the topic that should be created and
                subscribed to by the `adapter`.
            interface_handler (Ice.QObject handler.): iced-telemetry interface to
                be used as the subscription endpoint for the topic.
                
                		- `handlers`: A dictionary containing the Ice::Identity and
                Ice::Object references of the handler's implemented interfaces.
                		- `proxy`: The proxy object created by the `addWithUUID` method
                of the adapter.
                		- `topic_name`: The name of the topic to which the proxy is subscribed.
                		- `subscribe_done`: A boolean variable indicating whether the
                subscription to the topic has been successfully completed.
                		- `qos`: An Ice::QoS instance representing the quality of service
                (QoS) parameters for the subscription.

        Returns:
            Adapter` instance: an instance of an `Iceoryaka.Adapters.ObjectAdapter`.
            
            		- `adapter`: This is an object representing the adapter created by
            the function. It has methods for activation and deactivation of the
            adapter, as well as methods for adding and removing subscribers.
            		- `proxy`: This is a proxy object that represents the iceoryx
            connection point. It has methods for one-way communication with the proxy.
            		- `handler`: This is an object representing the interface handler
            associated with the adapter. It provides methods for registering and
            retrieving interfaces, as well as other operations related to the interface.
            		- `topic_name`: This is the name of the topic for which a subscriber
            was created.
            		- `subscribe_done`: This is a boolean flag indicating whether the
            subscribe operation was successful or not. If it is set to `True`,
            then the subscription was successful, otherwise, it failed.
            		- `qos`: This is an object representing the quality of service (QoS)
            parameters for the subscriber. It includes properties such as the
            priority, maximum messages delay, and minimum messages delay.
            
            	The function returns an adapter object, which is an instance of a
            class that inherits from the `iceoryx::adapter` class. This class
            provides methods for managing the adapter's life cycle, including
            activation and deactivation, as well as methods for adding and removing
            subscribers.

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
        Creates an Adaptor object using a pre-defined property name and adds an
        interface handler to it. It then activates the adaptor.

        Args:
            property_name ("identity".): property name of the Java class to be
                accessed through the ice-connector interface, which is used to
                create an object adapter and add it to the interfaces managed by
                the IceConnector instance.
                
                	1/ `property_name`: The input parameter is the name of an interface
                handler to be created as an adapter. It can be a string or any
                other object that supports the `str.lower()` method.
            interface_handler (str): handle of an interface object that provides
                services or resources to the adapter.

        """
        adapter = self.ice_connector.createObjectAdapter(property_name)
        adapter.add(interface_handler, self.ice_connector.stringToIdentity(property_name.lower()))
        adapter.activate()


class InterfaceManager:
    def __init__(self, ice_config_file):
        # TODO: Make ice connector singleton
        """
        Initialize the Ice framework, retrieve property values, and create instances
        of the Requires, Publishes, Implements, and Subscribes classes.

        Args:
            ice_config_file (file.): icy file that contains configuration settings
                for the given code, which is then passed to the `Ice.initialize()`
                method to initialize Ice components accordingly.
                
                		- `self.ice_config_file`: The file path or URL containing the
                configuration for an Ice environment.
                		- `self.ice_connector`: An initialized instance of the `Ice.Connector`
                class, which represents the connection to the Ice environment.
                		- `needs_rcnode`: A boolean value indicating whether the Ice
                environment requires a resource container node (RC node).
                		- `self.topic_manager`: An instance of the `TopicManager` class,
                which manages the topics published and subscribed by the application.
                If `needs_rcnode` is `True`, this attribute is set to `None`.
                		- `self.status`: An integer value representing the status of the
                Ice environment (e.g., `0` for normal operation, non-zero values
                for error conditions).
                		- `self.parameters`: A dictionary containing the properties of
                the Ice environment, where each key is a property name and each
                value is the property value. These properties include the connection
                string, authentication settings, and other configuration options.

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
        Converts a property value from an ice-connector instance into a `TopicManagerPrx`
        proxy object and returns it, or throws an `Ice.ConnectionRefusedException`
        if the given proxy cannot be connected to.

        Returns:
            IceStorm.TopicManagerPrx object: an instance of the `TopicManagerPrx`
            class, representing a proxy for the Topic Manager.
            
            		- `proxy`: A Proxy object that represents the TopicManager, which
            can be used to access the TopicManager's methods and properties.
            		- `obj`: The resulting IceStorm.TopicManagerPrx instance that
            represents the TopicManager interface.

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
        Updates `result` with the proxies maps from the required and published
        resources, respectively.

        Returns:
            dict: a dictionary of proxies for both required and published endpoints.

        """
        result = {}
        result.update(self.requires.get_proxies_map())
        result.update(self.publishes.get_proxies_map())
        return result

    def destroy(self):
        if self.ice_connector:
            self.ice_connector.destroy()




