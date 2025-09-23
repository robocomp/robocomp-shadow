import time
import Ice
import IceStorm
from rich.console import Console, Text
console = Console()

Ice.loadSlice("-I ./generated/ --all ./generated/Camera360RGB.ice")
import RoboCompCamera360RGB
Ice.loadSlice("-I ./generated/ --all ./generated/Lidar3D.ice")
import RoboCompLidar3D
Ice.loadSlice("-I ./generated/ --all ./generated/Lidar3DPub.ice")
import RoboCompLidar3DPub

class ImgType(list):
    def __init__(self, iterable=list()):
        super(ImgType, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, byte)
        super(ImgType, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, byte)
        super(ImgType, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, byte)
        super(ImgType, self).insert(index, item)
setattr(RoboCompCamera360RGB, "ImgType", ImgType)

class TCategories(list):
    def __init__(self, iterable=list()):
        super(TCategories, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, int)
        super(TCategories, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, int)
        super(TCategories, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, int)
        super(TCategories, self).insert(index, item)
setattr(RoboCompLidar3D, "TCategories", TCategories)

class TPoints(list):
    def __init__(self, iterable=list()):
        super(TPoints, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, RoboCompLidar3D.TPoint)
        super(TPoints, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, RoboCompLidar3D.TPoint)
        super(TPoints, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, RoboCompLidar3D.TPoint)
        super(TPoints, self).insert(index, item)
setattr(RoboCompLidar3D, "TPoints", TPoints)

class TFloatArray(list):
    def __init__(self, iterable=list()):
        super(TFloatArray, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, float)
        super(TFloatArray, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, float)
        super(TFloatArray, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, float)
        super(TFloatArray, self).insert(index, item)
setattr(RoboCompLidar3D, "TFloatArray", TFloatArray)

class TIntArray(list):
    def __init__(self, iterable=list()):
        super(TIntArray, self).__init__(iterable)

    def append(self, item):
        assert isinstance(item, int)
        super(TIntArray, self).append(item)

    def extend(self, iterable):
        for item in iterable:
            assert isinstance(item, int)
        super(TIntArray, self).extend(iterable)

    def insert(self, index, item):
        assert isinstance(item, int)
        super(TIntArray, self).insert(index, item)
setattr(RoboCompLidar3D, "TIntArray", TIntArray)


import lidar3dpubI

class Publishes:
    def __init__(self, ice_connector:Ice.CommunicatorI, topic_manager, parameters):
        self.ice_connector = ice_connector
        self.mprx={}
        self.topic_manager = topic_manager


    def create_topic(self, property_name, topic_name, prefix, ice_proxy):
        topic = False
        topic_full_name = f"{prefix}/{topic_name}" if prefix else topic_name

        while not topic:
            try:
                topic = self.topic_manager.retrieve(topic_full_name)
            except IceStorm.NoSuchTopic:
                try:
                    console.log(f"{Text('WARNING', style='yellow')} {topic_full_name} topic did not create. {Text('Creating...', style='green')}")
                    topic = self.topic_manager.create(topic_full_name)
                except:
                    console.log(f"{Text('WARNING', style='yellow')}publishing the {topic_full_name} topic. It is possible that other component have created")

        pub = topic.getPublisher().ice_oneway()
        proxy = ice_proxy.uncheckedCast(pub)
        self.mprx[property_name] = proxy
        return proxy

    def get_proxies_map(self):
        return self.mprx


class Requires:
    def __init__(self, ice_connector:Ice.CommunicatorI, parameters):
        self.ice_connector = ice_connector
        self.mprx={}

        self.Camera360RGB = self.create_proxy("Camera360RGB", RoboCompCamera360RGB.Camera360RGBPrx, parameters["Proxies"]["Camera360RGB"])

    def get_proxies_map(self):
        return self.mprx

    def create_proxy(self, property_name, ice_proxy, proxy_string):
        try:
            base_prx = self.ice_connector.stringToProxy(proxy_string)
            proxy = ice_proxy.uncheckedCast(base_prx)
            self.mprx[property_name] = proxy
            return True, proxy
        
        except Ice.Exception as e:
            console.print_exception(e)
            console.log(f'Cannot get {property_name} property.')
            self.status = -1
            return False, None


class Subscribes:
    def __init__(self, ice_connector:Ice.CommunicatorI, topic_manager, default_handler, parameters):
        self.ice_connector = ice_connector
        self.topic_manager = topic_manager

        self.Lidar3DPub = self.create_adapter("Lidar3DPub", parameters["Endpoints"]["Lidar3DPubPrefix"], 
                                                lidar3dpubI.Lidar3DPubI(default_handler, ""), parameters["Endpoints"]["Lidar3DPubTopic"])

    def create_adapter(self, topic_name, prefix, interface_handler, endpoint_string):
        topic_full_name = f"{prefix}/{topic_name}" if prefix else topic_name

        adapter = self.ice_connector.createObjectAdapterWithEndpoints(topic_full_name, endpoint_string)
        handler = interface_handler
        proxy = adapter.addWithUUID(handler).ice_oneway()
        subscribe_done = False
        while not subscribe_done:
            try:
                topic = self.topic_manager.retrieve(topic_full_name)
                subscribe_done = True
            except Ice.Exception as e:
                try:
                    console.log(f"{Text('WARNING', style='yellow')} {topic_full_name} topic did not create. {Text('Creating...', style='green')}")
                    topic = self.topic_manager.create(topic_full_name)
                    subscribe_done = True
                except:
                    print(f"{Text('WARNING', style='yellow')}publishing the {topic_full_name} topic. It is possible that other component have created")
        qos = {}
        topic.subscribeAndGetPublisher(qos, proxy)
        adapter.activate()
        return adapter


class Implements:
    def __init__(self, ice_connector:Ice.CommunicatorI, default_handler, parameters):
        self.ice_connector = ice_connector

    def create_adapter(self, property_name, interface_handler, endpoint_string):
        try:
            adapter = self.ice_connector.createObjectAdapterWithEndpoints(property_name, endpoint_string)
            adapter.add(interface_handler, self.ice_connector.stringToIdentity(property_name.lower()))
            adapter.activate()
            console.log(f"{property_name} adapter created in port {endpoint_string}")
        except:
            console.log(f"{Text('ERROR', style='red')} creating or activating adapter for{property_name}")
            self.status = -1


class InterfaceManager:
    def __init__(self, configData):

        init_data = Ice.InitializationData()
        init_data.properties = Ice.createProperties()
        init_data.properties.setProperty("Ice.Warn.Connections", configData["Ice"]["Warn"]["Connections"])
        init_data.properties.setProperty("Ice.Trace.Network", configData["Ice"]["Trace"]["Network"])
        init_data.properties.setProperty("Ice.Trace.Protocol", configData["Ice"]["Trace"]["Protocol"])
        init_data.properties.setProperty("Ice.MessageSizeMax", configData["Ice"]["MessageSizeMax"])
        self.ice_connector = Ice.initialize(init_data)

        self.status = 0

        needs_rcnode = True
        self.topic_manager = self.init_topic_manager(configData) if needs_rcnode else None

        self.requires = Requires(self.ice_connector, configData)
        self.publishes = Publishes(self.ice_connector, self.topic_manager, configData)
        self.implements = None
        self.subscribes = None

    def init_topic_manager(self, configData):
        obj = self.ice_connector.stringToProxy(configData["Proxies"]["TopicManager"])
        try:
            return IceStorm.TopicManagerPrx.checkedCast(obj)
        except Ice.ConnectionRefusedException as e:
            console.log(Text('Cannot connect to rcnode! This must be running to use pub/sub.', 'red'))
            self.status = -1
            exit(-1)

    def set_default_hanlder(self, handler, configData):
        self.implements = Implements(self.ice_connector, handler, configData)
        self.subscribes = Subscribes(self.ice_connector, self.topic_manager, handler, configData)

    def get_proxies_map(self):
        result = {}
        result.update(self.requires.get_proxies_map())
        result.update(self.publishes.get_proxies_map())
        return result

    def destroy(self):
        if self.ice_connector:
            self.ice_connector.destroy()




