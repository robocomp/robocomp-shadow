/*
 *    Copyright (C) 2023 by YOUR NAME HERE
 *
 *    This file is part of RoboComp
 *
 *    RoboComp is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    RoboComp is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with RoboComp.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "specificworker.h"

#pragma region Robocomp Methods

/**
* \brief Default constructor
*/
SpecificWorker::SpecificWorker(TuplePrx tprx, bool startup_check) : GenericWorker(tprx)
{
    this->startup_check_flag = startup_check;
    // Uncomment if there's too many debug messages
    // but it removes the possibility to see the messages
    // shown in the console with qDebug()
	QLoggingCategory::setFilterRules("*.debug=false\n");
}

/**
* \brief Default destructor
*/
SpecificWorker::~SpecificWorker()
{
    std::cout << "Destroying SpecificWorker" << std::endl;

    //close output stream
    file.close();
    std::cout << "Metric File closed" << std::endl;

    if(robot)
        delete robot;
}

bool SpecificWorker::setParams(RoboCompCommonBehavior::ParameterList params)
{
    // Save locale setting
    const std::string oldLocale=std::setlocale(LC_NUMERIC,nullptr);
    // Force '.' as the radix point. If you comment this out,
    // you'll get output similar to the OP's GUI mode sample
    std::setlocale(LC_NUMERIC,"C");
    try
    {
        pars.delay = params.at("delay").value == "true" or (params.at("delay").value == "True");
        pars.do_joystick = params.at("do_joystick").value == "true" or (params.at("do_joystick").value == "True");
    }
    catch (const std::exception &e)
    {std::cout <<"Error reading the config \n" << e.what() << std::endl << std::flush; }

    // Restore locale setting
    std::setlocale(LC_NUMERIC,oldLocale.c_str());
	try
	{
		agent_name = params.at("agent_name").value;
		agent_id = stoi(params.at("agent_id").value);
		tree_view = params.at("tree_view").value == "true";
		graph_view = params.at("graph_view").value == "true";
		qscene_2d_view = params.at("2d_view").value == "true";
		osg_3d_view = params.at("3d_view").value == "true";
	}
	catch(const std::exception &e){ std::cout << e.what() << " Error reading params from config file" << std::endl;};

    return true;
}

void SpecificWorker::initialize(int period)
{
    std::cout << "Initialize worker" << std::endl;
    this->Period = period;
    if(this->startup_check_flag)
    {
        this->startup_check();
    }
    else
    {

//        //Initialize stream to .csv file
//        std::string filename = "data.csv";
//        file.open(filename);
//        //if file is open
//        if(file.is_open())
//        {
//            //Pint message
//            std::cout << "File open" << std::endl;
//            //write header
//            file << "experiment_ID" << ";" << "timestamp" << ";" << "Person_pose_x" << ";" << "Person_pose_y" << ";"
//            << "object_pose_x" << ";" << "object_pose_y" << ";" << "person_target_x" << ";" << "person_target_y"
//            << ";" << "robot_pose_x" << ";" << "robot_pose_y" << ";" << "Collision_Edge" << ";" << "Node_Avoid"
//            <<";" << "avoid_target_x" <<";" << "avoid_target_y" <<std::endl;
//        }
//        else
//        {
//            std::cout << "Error opening file" << std::endl;
//        }

        robot = new webots::Supervisor();

        // Inicializa los motores y los sensores de posición.
        const char *motorNames[4] = {"wheel2", "wheel1", "wheel4", "wheel3"};
        //const char *sensorNames[4] = {"wheel1sensor", "wheel2sensor", "wheel3sensor", "wheel4sensor"};
        qInfo() << 1;
        // Inicializa los sensores soportados.
        lidar_helios = robot->getLidar("helios");
        lidar_pearl = robot->getLidar("bpearl");
        camera = robot->getCamera("camera");
        range_finder = robot->getRangeFinder("range-finder");
        camera360_1 = robot->getCamera("camera_360_1");
        camera360_2 = robot->getCamera("camera_360_2");

        // Inicializa el teclado.
        keyboard = robot->getKeyboard();

        this->Period = 1;

        // Activa los componentes en la simulación si los detecta.
        if(lidar_helios) lidar_helios->enable(this->Period);
        if(lidar_pearl) lidar_pearl->enable(this->Period);
        if(camera) camera->enable(this->Period);
        if(range_finder) range_finder->enable(this->Period);
        if(camera360_1 && camera360_2){
            camera360_1->enable(this->Period);
            camera360_2->enable(this->Period);
        }
        for (int i = 0; i < 4; i++)
        {
            motors[i] = robot->getMotor(motorNames[i]);
            ps[i] = motors[i]->getPositionSensor();
            ps[i]->enable(this->Period);
            motors[i]->setPosition(INFINITY); // Modo de dad.
            motors[i]->setVelocity(0);
        }
        if(keyboard) keyboard->enable(this->Period);

        robotStartPosition[0] = robot->getFromDef("shadow")->getField("translation")->getSFVec3f()[0];
        robotStartPosition[1] = robot->getFromDef("shadow")->getField("translation")->getSFVec3f()[1];
        robotStartPosition[2] = robot->getFromDef("shadow")->getField("translation")->getSFVec3f()[2];

        qInfo () << "Robot start position: " << robotStartPosition[0] << " " << robotStartPosition[1] << " " << robotStartPosition[2];

        robotStartOrientation[0] = robot->getFromDef("shadow")->getField("rotation")->getSFRotation()[0];
        robotStartOrientation[1] = robot->getFromDef("shadow")->getField("rotation")->getSFRotation()[1];
        robotStartOrientation[2] = robot->getFromDef("shadow")->getField("rotation")->getSFRotation()[2];
        robotStartOrientation[3] = robot->getFromDef("shadow")->getField("rotation")->getSFRotation()[3];
        qInfo () << "Robot start orientation: " << robotStartOrientation[0] << " " << robotStartOrientation[1] << " " << robotStartOrientation[2] << " " << robotStartOrientation[3];

        Eigen::Vector3d axis(robotStartOrientation[0], robotStartOrientation[1], robotStartOrientation[2]);
        Eigen::AngleAxisd angleAxis(robotStartOrientation[3], axis.normalized());
        Eigen::Quaterniond quaternion(angleAxis);

        Eigen::Vector3d eulerAngles = quaternion.toRotationMatrix().eulerAngles(0, 1, 2);

//        parseHumanObjects(true);
//        setObstacleStartPosition();
        webots_initializated = true;

//        //METRICS CSV
//        std::string filename = "experiment_data.csv";
//        file.ope

		// create graph

        G = std::make_shared<DSR::DSRGraph>(0, agent_name, agent_id, "shadow_scene.json"); // Init nodes
		std::cout<< __FUNCTION__ << "Graph loaded" << std::endl;

        // if (auto robot_node_ = G->get_node("Shadow"); robot_node_.has_value())
        // {
        //     auto robot_node = robot_node_.value();
        //     G->add_or_modify_attrib_local<robot_start_position_att>(robot_node, std::vector<float> {robotStartPosition[0], robotStartPosition[1], robotStartPosition[2]});
        //     G->add_or_modify_attrib_local<robot_start_orientation_att>(robot_node, std::vector<float> {0.f, 0.f, eulerAngles.z()});
        //     G->update_node(robot_node);
        // }

		//dsr update signals
		connect(G.get(), &DSR::DSRGraph::update_node_signal, this, &SpecificWorker::modify_node_slot);
		//connect(G.get(), &DSR::DSRGraph::update_edge_signal, this, &SpecificWorker::modify_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::update_node_attr_signal, this, &SpecificWorker::modify_node_attrs_slot);
		//connect(G.get(), &DSR::DSRGraph::update_edge_attr_signal, this, &SpecificWorker::modify_edge_attrs_slot);
		//connect(G.get(), &DSR::DSRGraph::del_edge_signal, this, &SpecificWorker::del_edge_slot);
		//connect(G.get(), &DSR::DSRGraph::del_node_signal, this, &SpecificWorker::del_node_slot);

		// Graph viewer
		using opts = DSR::DSRViewer::view;
		int current_opts = 0;
		opts main = opts::none;
		if(tree_view)
		{
		    current_opts = current_opts | opts::tree;
		}
		if(graph_view)
		{
		    current_opts = current_opts | opts::graph;
		    main = opts::graph;
		}
		if(qscene_2d_view)
		{
		    current_opts = current_opts | opts::scene;
		}
		if(osg_3d_view)
		{
		    current_opts = current_opts | opts::osg;
		}
		graph_viewer = std::make_unique<DSR::DSRViewer>(this, G, current_opts, main);
		setWindowTitle(QString::fromStdString(agent_name + "-") + QString::number(agent_id));

		/***
		Custom Widget
		In addition to the predefined viewers, Graph Viewer allows you to add various widgets designed by the developer.
		The add_custom_widget_to_dock method is used. This widget can be defined like any other Qt widget,
		either with a QtDesigner or directly from scratch in a class of its own.
		The add_custom_widget_to_dock method receives a name for the widget and a reference to the class instance.
		***/
		//graph_viewer->add_custom_widget_to_dock("CustomWidget", &custom_widget);

        hide();
        //Insert in the config?
        timer.start(Period);
    }

}

void SpecificWorker::compute()
{
    if(reset)
    {
        qInfo() << "Resetting the simulation.";
        setElementsToStartPosition();
        reset = false;
    }
    // Getting the data from simulation.
    if(lidar_helios) receiving_lidarData("helios", lidar_helios, double_buffer_helios,  helios_delay_queue);
    if(lidar_pearl) receiving_lidarData("bpearl", lidar_pearl, double_buffer_pearl, pearl_delay_queue);
    if(camera) receiving_cameraRGBData(camera);
    if(range_finder) receiving_depthImageData(range_finder);
    if(camera360_1 && camera360_2) receiving_camera360Data(camera360_1, camera360_2);

    auto std_model_name = robot->getFromDef("WALLS")->getField("url")->getMFString(0);
    std::cout << "Model name: " << std_model_name << std::endl;
    auto root_node_ = G->get_node("root");
    if(not root_node_.has_value())
    {
        std::cout << "Root node not found" << std::endl;
        return;
    }
    auto root_node = root_node_.value();
    G->add_or_modify_attrib_local<path_att>(root_node, std::string(std_model_name));
    G->update_node(root_node);

    if(auto path_name = G->get_attrib_by_name<path_att >(root_node); path_name.has_value())
    {
        std::cout << "Path name: " << path_name.value().get() << std::endl;
    }

    insert_robot_speed_dsr();
//    if(keyboard)
//    {
//        int key = keyboard->getKey();
//
//        if ((key >= 0) && key != previousKey) {
//            switch (key) {
//                case 'R':
//                    // Print R key pressedr
//                    setElementsToStartPosition();
//                    break;
//            }
//        }
//        previousKey = key;
//    }

//    robot->step(this->Period);
    robot->step(33);
//    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now).count() << std::endl;
//
//    parseHumanObjects();


//    //TODO: DELETE, only for debuggin purpose
//    //Create 15 equispace points Robocomp gridder path between (1000, 0) and (1000, 3000) points
//    RoboCompGridder::TPath path;
//    for(int i = 0; i < 15; i++)
//    {
//        RoboCompGridder::TPoint point;
//        point.x = 1000;
//        point.y = i * 200;
//        path.push_back(point);
//        //print point
//        std::cout << "Point: " << point.x << " " << point.y << std::endl;
//    }
//    //transform path using setPathToHuman

//    humansMovement();


    //            file << "experiment_ID" << ";" << "timestamp" << ";" << "Person_pose_x" << ";" << "Person_pose_y" << ";"
    //            << "object_pose_x" << ";" << "object_pose_y" << ";" << "person_target_x" << ";" << "person_target_y"
    //            << ";" << "robot_pose_x" << ";" << "robot_pose_y" << ";" << "Collision_Edge" << ";" << "Node_Avoid"
    //            <<";" << "avoid_target_x" <<";" << "avoid_target_y" <<std::endl;

    //get webots metric data
//    auto webots_data = getPositions();
//    //get DSR metrics
//    auto dsr_data = calculate_collision_metrics();
//    auto avoid_target = std::get<2>(dsr_data);
//    file << std::to_string(experiment_id) << ";" << std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()) //ID , TIMESTAMP
//    << ";" << std::to_string(webots_data[0][0]) << ";" << std::to_string(webots_data[0][1]) << ";" << std::to_string(webots_data[1][0]) << ";" << std::to_string(webots_data[1][1]) // Person_x, Person_y, object_x, object_y
//    << ";" << std::to_string(webots_data[2][0]) << ";" << std::to_string(webots_data[2][1]) << ";" << std::to_string(webots_data[3][0]) << ";" << std::to_string(webots_data[3][1]) // Person_target_x, Person_target_y, robot_pose_x, robot_pose_y
//    << ";" << std::to_string(std::get<0>(dsr_data))<< ";"<< std::to_string(std::get<1>(dsr_data))<< ";"<< std::to_string(avoid_target.x())<< ";" <<std::to_string(avoid_target.y())<<std::endl; //Collision_Edge, Node_Avoid, avoid_target_x, avoid_target_y

    fps.print("FPS:");
}

int SpecificWorker::startup_check()
{
    std::cout << "Startup check" << std::endl;
    QTimer::singleShot(200, qApp, SLOT(quit()));
    return 0;
}

#pragma endregion Robocomp Methods

#pragma region Data-Catching Methods

void SpecificWorker::receiving_camera360Data(webots::Camera* _camera1, webots::Camera* _camera2)
{
    RoboCompCamera360RGB::TImage newImage360;

    // Aseguramos de que ambas cámaras tienen la misma resolución, de lo contrario, deberás manejar las diferencias.
    if (_camera1->getWidth() != _camera2->getWidth() || _camera1->getHeight() != _camera2->getHeight())
    {
        std::cerr << "Error: Cameras with different resolutions." << std::endl;
        return;
    }

    // Timestamp calculation
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    newImage360.alivetime = millis;

    // La resolución de la nueva imagen será el doble en el ancho ya que estamos combinando las dos imágenes.
    newImage360.width = 2 * _camera1->getWidth();
    newImage360.height = _camera1->getHeight();

    // Establecer el periodo de refresco de la imagen en milisegundos.
//    newImage360.period = this->Period;

    // Establecer el periodo real del compute de refresco de la imagen en milisegundos.
    newImage360.period = fps.get_period();

    const unsigned char* webotsImageData1 = _camera1->getImage();
    const unsigned char* webotsImageData2 = _camera2->getImage();
    cv::Mat img_1 = cv::Mat(cv::Size(_camera1->getWidth(), _camera1->getHeight()), CV_8UC4);
    cv::Mat img_2 = cv::Mat(cv::Size(_camera2->getWidth(), _camera2->getHeight()), CV_8UC4);
    img_1.data = (uchar *)webotsImageData1;
    cv::cvtColor(img_1, img_1, cv::COLOR_RGBA2RGB);
    img_2.data = (uchar *)webotsImageData2;
    cv::cvtColor(img_2, img_2, cv::COLOR_RGBA2RGB);
    cv::Mat img_final = cv::Mat(cv::Size(_camera1->getWidth()*2, _camera1->getHeight()), CV_8UC3);
    img_1.copyTo(img_final(cv::Rect(0, 0, _camera1->getWidth(), _camera1->getHeight())));
    img_2.copyTo(img_final(cv::Rect(_camera1->getWidth(), 0, _camera1->getWidth(), _camera2->getHeight())));

    // Asignar la imagen RGB 360 al tipo TImage de Robocomp
    newImage360.image.resize(img_final.total()*img_final.elemSize());
    memcpy(&newImage360.image[0], img_final.data, img_final.total()*img_final.elemSize());

    //newImage360.image = rgbImage360;
    newImage360.compressed = false;

    if(pars.delay)
        camera_queue.push(newImage360);

    // Asignamos el resultado final al atributo de clase (si tienes uno).
    double_buffer_360.put(std::move(newImage360));

    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - now).count() << std::endl;
}

void SpecificWorker::receiving_lidarData(string name, webots::Lidar* _lidar, DoubleBuffer<RoboCompLidar3D::TData, RoboCompLidar3D::TData> &_lidar3dData, FixedSizeDeque<RoboCompLidar3D::TData>& delay_queue)
{
    if (!_lidar) { std::cout << "No lidar available." << std::endl; return; }

    const float *rangeImage = _lidar->getRangeImage();
    int horizontalResolution = _lidar->getHorizontalResolution();
    int verticalResolution = _lidar->getNumberOfLayers();
    double minRange = _lidar->getMinRange();
    double maxRange = _lidar->getMaxRange();
    double fov = _lidar->getFov();
    double verticalFov = _lidar->getVerticalFov();

    // Timestamp calculation
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

    // Configuration settings
    RoboCompLaser::TLaserData newLaserData;
    RoboCompLaser::LaserConfData newLaserConfData;
    RoboCompLidar3D::TData newLidar3dData;

    // General Lidar values
    newLidar3dData.timestamp = millis;
    newLidar3dData.period = fps.get_period();
    newLaserConfData.maxDegrees = fov;
    newLaserConfData.maxRange = maxRange;
    newLaserConfData.minRange = minRange;
    newLaserConfData.angleRes = fov / horizontalResolution;

    //std::cout << "horizontal resolution: " << horizontalResolution << " vertical resolution: " << verticalResolution << " fov: " << fov << " vertical fov: " << verticalFov << std::endl;

    if(!rangeImage) { std::cout << "Lidar data empty." << std::endl; return; }


    for (int j = 0; j < verticalResolution; ++j) {
        for (int i = 0; i < horizontalResolution; ++i) {
            int index = j * horizontalResolution + i;

            //distance meters to millimeters
            const float distance = rangeImage[index]; //Meters

            //TODO rotacion del eje y con el M_PI, solucionar
            float horizontalAngle = M_PI - i * newLaserConfData.angleRes - fov / 2;

            if(name == "helios")
            {
                verticalFov = 2.8;
            }

            float verticalAngle = M_PI + j * (verticalFov / verticalResolution) - verticalFov / 2;

            //Calculate Cartesian co-ordinates and rectify axis positions
            Eigen::Vector3f lidar_point(
                    distance * cos(horizontalAngle) * cos(verticalAngle),
                    distance * sin(horizontalAngle) * cos(verticalAngle),
                    distance * sin(verticalAngle));

            if (not (std::isinf(lidar_point.x()) or std::isinf(lidar_point.y()) or std::isinf(lidar_point.z())))
            {
                if (not (name == "bpearl" and lidar_point.z() < 0) and
                    not (name == "helios" and (verticalAngle > 4.10152 or verticalAngle <2.87979)))//down limit+, uper limit-, horizon line is PI
//                        not (name == "helios" and (verticalAngle > 2.5307*1.5 or verticalAngle <1.309*1.5 )))
                    //not (name == "helios" and false))
                {
                    RoboCompLidar3D::TPoint point;

                    point.x = lidar_point.x();
                    point.y = lidar_point.y();
                    point.z = lidar_point.z();

                    point.r = lidar_point.norm();  // distancia radial
                    point.phi = horizontalAngle;  // ángulo horizontal // -x para hacer [PI, -PI] y no [-PI, PI]
                    point.theta = verticalAngle;  // ángulo vertical
                    point.distance2d = std::hypot(lidar_point.x(),lidar_point.y());  // distancia en el plano xy

                    RoboCompLaser::TData data;
                    data.angle = point.phi;
                    data.dist = point.distance2d;

                    newLidar3dData.points.push_back(point);
                    newLaserData.push_back(data);
                }
            }
        }
    }
    //Points order to angles
    std::ranges::sort(newLidar3dData.points, {}, &RoboCompLidar3D::TPoint::phi);

    laserData = newLaserData;
    laserDataConf = newLaserConfData;

    //Is it necessary to use two lidar queues? One for each lidaR?
    if(pars.delay)
        delay_queue.push(newLidar3dData);

    _lidar3dData.put(std::move(newLidar3dData));
}
void SpecificWorker::receiving_cameraRGBData(webots::Camera* _camera){
    RoboCompCameraRGBDSimple::TImage newImage;

    // Se establece el periodo de refresco de la imagen en milisegundos.
//    newImage.period = this->Period;
    newImage.period = fps.get_period();

    // Timestamp calculation
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    newImage.alivetime = millis;

    // Obtener la resolución de la imagen.
    newImage.width = _camera->getWidth();
    newImage.height = _camera->getHeight();

    const unsigned char* webotsImageData = _camera->getImage();

    // Crear un vector para la nueva imagen RGB.
    std::vector<unsigned char> rgbImage;
    rgbImage.reserve(3 * newImage.width * newImage.height);  // Reservar espacio para RGB

    for (int y = 0; y < newImage.height; y++)
    {
        for (int x = 0; x < newImage.width; x++)
        {
            // Extraer cada canal por separado
            unsigned char r = _camera->imageGetRed(webotsImageData, newImage.width, x, y);
            unsigned char g = _camera->imageGetGreen(webotsImageData, newImage.width, x, y);
            unsigned char b = _camera->imageGetBlue(webotsImageData, newImage.width, x, y);

            // Añadir los canales al vector BGR final.
            rgbImage.push_back(b);
            rgbImage.push_back(g);
            rgbImage.push_back(r);
        }
    }

    // Asignar la imagen RGB al tipo TImage de Robocomp
    newImage.image = rgbImage;
    newImage.compressed = false;

    // Asignamos el resultado final al atributo de clase
    this->cameraImage = newImage;
}
void SpecificWorker::receiving_depthImageData(webots::RangeFinder* _rangeFinder){
    RoboCompCameraRGBDSimple::TDepth newDepthImage;

    // Se establece el periodo de refresco de la imagen en milisegundos.
    newDepthImage.period = fps.get_period();

    // Obtener la resolución de la imagen de profundidad.
    newDepthImage.width = _rangeFinder->getWidth();
    newDepthImage.height = _rangeFinder->getHeight();
    newDepthImage.depthFactor = _rangeFinder->getMaxRange();
    newDepthImage.compressed = false;

    // Obtener la imagen de profundidad
    const float* webotsDepthData = _rangeFinder->getRangeImage();

    // Accedemos a cada depth value y le aplicamos un factor de escala.
    const int imageElementCount = newDepthImage.width * newDepthImage.height;

    for(int i= 0 ; i<imageElementCount; i++){

        // Este es el factor de escala a aplicar.
        float scaledValue = webotsDepthData[i] * 10;

        // Convertimos de float a array de bytes.
        unsigned char singleElement[sizeof(float)];
        memcpy(singleElement, &scaledValue, sizeof(float));

        for(uint j=0; j<sizeof(float); j++){
            newDepthImage.depth.emplace_back(singleElement[j]);
        }
    }

    // Asignamos el resultado final al atributo de clase
    this->depthImage = newDepthImage;
}

#pragma endregion Data-Catching Methods

#pragma region CameraRGBDSimple

RoboCompCameraRGBDSimple::TRGBD SpecificWorker::CameraRGBDSimple_getAll(std::string camera)
{
    RoboCompCameraRGBDSimple::TRGBD newRGBD;

    newRGBD.image = this->cameraImage;
    newRGBD.depth = this->depthImage;
    // TODO: Que devuelva tambien la nube de puntos.

    return newRGBD;
}

RoboCompCameraRGBDSimple::TDepth SpecificWorker::CameraRGBDSimple_getDepth(std::string camera)
{
    return this->depthImage;
}

RoboCompCameraRGBDSimple::TImage SpecificWorker::CameraRGBDSimple_getImage(std::string camera)
{
    return this->cameraImage;
}

RoboCompCameraRGBDSimple::TPoints SpecificWorker::CameraRGBDSimple_getPoints(std::string camera)
{
    printNotImplementedWarningMessage("CameraRGBDSimple_getPoints");
    return RoboCompCameraRGBDSimple::TPoints();
}

#pragma endregion CamerRGBDSimple

#pragma region Lidar

RoboCompLaser::TLaserData SpecificWorker::Laser_getLaserAndBStateData(RoboCompGenericBase::TBaseState &bState)
{
    return laserData;
}

RoboCompLaser::LaserConfData SpecificWorker::Laser_getLaserConfData()
{
    return laserDataConf;
}

RoboCompLaser::TLaserData SpecificWorker::Laser_getLaserData()
{
    return laserData;
}

RoboCompLidar3D::TData SpecificWorker::Lidar3D_getLidarData(std::string name, float start, float len, int decimationDegreeFactor)
{
    if(name == "helios") {
        if(pars.delay && helios_delay_queue.full())
            return helios_delay_queue.back();
        else
            return double_buffer_helios.get_idemp();
    }
    else if(name == "bpearl")
    {
        if(pars.delay && pearl_delay_queue.full())
            return pearl_delay_queue.back();
        else
            return double_buffer_pearl.get_idemp();
    }
    else
    {
        cout << "Getting data from a not implemented lidar (" << name << "). Try 'helios' or 'pearl' instead." << endl;
        return RoboCompLidar3D::TData();
    }
}

RoboCompLidar3D::TData SpecificWorker::Lidar3D_getLidarDataWithThreshold2d(std::string name, float distance, int decimationDegreeFactor)
{
    printNotImplementedWarningMessage("Lidar3D_getLidarDataWithThreshold2d");
    return RoboCompLidar3D::TData();
}

RoboCompLidar3D::TDataImage SpecificWorker::Lidar3D_getLidarDataArrayProyectedInImage(std::string name)
{
    printNotImplementedWarningMessage("Lidar3D_getLidarDataArrayProyectedInImage");
    return RoboCompLidar3D::TDataImage();
}

#pragma endregion Lidar

#pragma region Camera360

RoboCompCamera360RGB::TImage SpecificWorker::Camera360RGB_getROI(int cx, int cy, int sx, int sy, int roiwidth, int roiheight)
{
    if(pars.delay)
    {
        if(camera_queue.full())
            return camera_queue.back();
    }

    return double_buffer_360.get_idemp();
}

#pragma endregion Camera360


#pragma region OmniRobot

void SpecificWorker::OmniRobot_correctOdometer(int x, int z, float alpha)
{
    printNotImplementedWarningMessage("OmniRobot_correctOdometer");
}

void SpecificWorker::OmniRobot_getBasePose(int &x, int &z, float &alpha)
{
    printNotImplementedWarningMessage("OmniRobot_getBasePose");
}

void SpecificWorker::OmniRobot_getBaseState(RoboCompGenericBase::TBaseState &state)
{
    webots::Node *robotNode = robot->getFromDef("shadow");

    state.x = robotNode->getField("translation")->getSFVec3f()[0];
    state.z = robotNode->getField("translation")->getSFVec3f()[1];
    state.alpha = robotNode->getField("rotation")->getSFRotation()[0];
}

void SpecificWorker::OmniRobot_resetOdometer()
{
    printNotImplementedWarningMessage("OmniRobot_resetOdometer");
}

void SpecificWorker::OmniRobot_setOdometer(RoboCompGenericBase::TBaseState state)
{
    printNotImplementedWarningMessage("OmniRobot_setOdometer");
}

void SpecificWorker::OmniRobot_setOdometerPose(int x, int z, float alpha)
{
    printNotImplementedWarningMessage("OmniRobot_setOdometerPose");
}

void SpecificWorker::OmniRobot_setSpeedBase(float advx, float advz, float rot)
{
    double speeds[4];

    advz *= 0.001;
    advx *= 0.001;

    speeds[0] = 1.0 / WHEEL_RADIUS * (advz + advx + (LX + LY) * rot);
    speeds[1] = 1.0 / WHEEL_RADIUS * (advz - advx - (LX + LY) * rot);
    speeds[2] = 1.0 / WHEEL_RADIUS * (advz - advx + (LX + LY) * rot);
    speeds[3] = 1.0 / WHEEL_RADIUS * (advz + advx - (LX + LY) * rot);
    printf("Speeds: vx=%.2f[m/s] vy=%.2f[m/s] ω=%.2f[rad/s]\n", advx, advz, rot);
    if(webots_initializated)
        for (int i = 0; i < 4; i++)
           motors[i]->setVelocity(speeds[i]);
}

void SpecificWorker::OmniRobot_stopBase()
{
    for (int i = 0; i < 4; i++)
    {
        motors[i]->setVelocity(0);
    }
}

#pragma endregion OmniRobot

#pragma region JoystickAdapter

//SUBSCRIPTION to sendData method from JoystickAdapter interface
void SpecificWorker::JoystickAdapter_sendData(RoboCompJoystickAdapter::TData data)
{
    // Declaration of the structure to be filled
    float side=0, adv=0, rot=0;
    /*
    // Iterate through the list of buttons in the data structure
    for (RoboCompJoystickAdapter::ButtonParams button : data.buttons) {
        // Currently does nothing with the buttons
    }
    */

    // Iterate through the list of axes in the data structure
    for (RoboCompJoystickAdapter::AxisParams axis : data.axes)
    {
        // Process the axis according to its name
        if(axis.name == "rotate")
            rot = axis.value;
        else if (axis.name == "advance")
            adv = axis.value;
        else if (axis.name == "side")
            side = axis.value;
        else
            cout << "[ JoystickAdapter ] Warning: Using a non-defined axes (" << axis.name << ")." << endl;
    }
    // Print the values of the speeds
    cout << "Side: " << side << " Advance: " << adv << " Rotate: " << rot << endl;
     if(pars.do_joystick)
         OmniRobot_setSpeedBase(side, adv, rot);
}

#pragma endregion JoystickAdapter

#pragma region DSR

void SpecificWorker::insert_robot_speed_dsr()
{
    auto shadow_velocity = robot->getFromDef("shadow")->getVelocity();

    if (auto robot_node_ = G->get_node("Shadow"); robot_node_.has_value())
    {
        auto robot_node = robot_node_.value();

//        auto rot = robot->getFromDef("shadow")->getField("rotation")->getSFRotation();
//
//        Eigen::Vector3d axis(rot[0], rot[1], rot[2]);
//        Eigen::AngleAxisd angleAxis(rot[3], axis.normalized());
//        Eigen::Quaterniond quaternion(angleAxis);
//
//        Eigen::Vector3d eulerAngles = quaternion.toRotationMatrix().eulerAngles(0, 1, 2);

        auto shadow_orientation = robot->getFromDef("shadow")->getOrientation();
        float orientation = atan2(shadow_orientation[1], shadow_orientation[0]) - M_PI_2;

        Eigen::Matrix2f rt_rotation_matrix;
        rt_rotation_matrix << cos(orientation), -sin(orientation),
                                sin(orientation), cos(orientation);

        // Multiply the velocity vector by the inverse of the rotation matrix to get the velocity in the robot reference system
        Eigen::Vector2f shadow_velocity_2d(shadow_velocity[1], shadow_velocity[0]);
        Eigen::Vector2f rt_rotation_matrix_inv = rt_rotation_matrix.inverse() * shadow_velocity_2d;


        //std::cout <<  "X speed " << shadow_velocity[1] << "; Y speed " << shadow_velocity[0] << "; Angular Speed " << shadow_velocity[5] << "; GLOBAL." << std::endl;
        //std::cout <<  "X speed " << rt_rotation_matrix_inv(0) << "; Y speed " << rt_rotation_matrix_inv(1) << "; Angular Speed " << shadow_velocity[5] << "; ROBOT REFERENCE SYSTEM." << std::endl;
//        std::cout <<  "X speed " << rt_rotation_matrix_inv(0,2) << "; Y speed " << rt_rotation_matrix_inv(1,2) << "; Angular Speed " << shadow_velocity[5] << "; ROBOT REFERENCE SYSTEM." << std::endl;

        // Velocidades puras en mm/s y rad/s
        double velocidad_x = 0.05; // Ejemplo: 100 mm/s
        double velocidad_y = 0.05; // Ejemplo: 150 mm/s
        double alpha = 0.075; // Ejemplo: 0.05 rad/s

        // Desviación estándar del ruido (ejemplo: 5% del valor de las velocidades)
        double ruido_stddev_x = 0.05 * velocidad_x;
        double ruido_stddev_y = 0.05 * velocidad_y;
        double ruido_stddev_alpha = 0.05 * alpha;

        G->add_or_modify_attrib_local<robot_current_advance_speed_att>(robot_node, (float)  (-rt_rotation_matrix_inv(0) + generarRuido(ruido_stddev_x)));
        G->add_or_modify_attrib_local<robot_current_side_speed_att>(robot_node, (float) (-rt_rotation_matrix_inv(1) + generarRuido(ruido_stddev_y)));
        G->add_or_modify_attrib_local<robot_current_angular_speed_att>(robot_node, (float) (shadow_velocity[5] + generarRuido(ruido_stddev_alpha)));
        auto now_aux = std::chrono::system_clock::now();
        auto epoch_aux = now_aux.time_since_epoch();
        auto milliseconds_aux = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_aux).count();
        G->add_or_modify_attrib_local<timestamp_alivetime_att>(robot_node, (uint64_t) milliseconds_aux);
        G->update_node(robot_node);
    }
}
// Función para generar ruido normal (media = 0, desviación estándar = stddev)
double SpecificWorker::generarRuido(double stddev)
{
    std::random_device rd; // Obtiene una semilla aleatoria del hardware
    std::mt19937 gen(rd()); // Generador de números aleatorios basado en Mersenne Twister
    std::normal_distribution<> d(0, stddev); // Distribución normal con media 0 y desviación estándar stddev
    return d(gen);
}
#pragma endregion DSR

void SpecificWorker::parseHumanObjects(bool firstTime) 
{
    webots::Node* crowdNode = robot->getFromDef("CROWD");
    webots::Field* childrenField = crowdNode->getFieldByIndex(0);
    for (int i = 0; i < childrenField->getCount(); ++i)
    {
        std::string nodeDEF = childrenField->getMFNode(i)->getDef();
        if(nodeDEF.find("HUMAN_") != std::string::npos)
        {
            humanObjects[i].node = childrenField->getMFNode(i);
            if(firstTime)
            {
                auto person_pose = humanObjects[i].node->getPosition();
                for (int j = 0; j < 2; j++)
                    humanObjects[i].startPosition.push_back(person_pose[j]);
                auto person_orientation = humanObjects[i].node->getOrientation();
                // Iterate over the orientation array and save the values in the startOrientation array
                for (int j = 0; j < 4; j++)
                    humanObjects[i].startOrientation.push_back(person_orientation[j]);
            }
        }
    }
}

RoboCompVisualElements::TObjects SpecificWorker::VisualElements_getVisualObjects(RoboCompVisualElements::TObjects objects)
{
    RoboCompVisualElements::TObjects objectsList;

    for (const auto &entry : humanObjects) {
        RoboCompVisualElements::TObject object;

        int id = entry.first;
        webots::Node *node = entry.second.node;
        const double *position = node->getPosition();

        object.id = id;
        object.x = position[0];
        object.y = position[1];

        objectsList.objects.push_back(object);
    }
    
    return objectsList;
}

void SpecificWorker::VisualElements_setVisualObjects(RoboCompVisualElements::TObjects objects)
{
    // Implement CODE
}

#pragma region Webots2Robocomp Methods

void SpecificWorker::humansMovement()
{
    if(!humanObjects.empty())
        for (auto& human : humanObjects)
        {
//            qInfo() << "ID" << human.first;

//            qInfo() << "Moving human " << human.first << human.second.path.size();
            moveHumanToNextTarget(human.first);

        }
}

void SpecificWorker::moveHumanToNextTarget(int humanId)
{
    webots::Node *humanNode = humanObjects[humanId].node;

    if(humanNode == nullptr)
    {
        qInfo() << "Human not found";
        return;
    }
        humanObjects[humanId].node->getPosition();
        
    double velocity[6];

    if(humanObjects[humanId].path.empty())
    {
//        qInfo() <<"Set velocity to 0, path empty";
        velocity[0] = 0.f;
        velocity[1] = 0.f;
        velocity[2] = 0.f;
        velocity[3] = 0.f;
        velocity[4] = 0.f;
        velocity[5] = 0.f;
    }
    else
    {
        const double *position = humanNode->getPosition();

        Eigen::Vector3d currentTarget {humanObjects[humanId].path.front().x - position[0], humanObjects[humanId].path.front().y - position[1] , 0};

        //Erase path if .norm < d = 300mm
        if(currentTarget.norm() < 0.3)
            humanObjects[humanId].path.erase(humanObjects[humanId].path.begin());

        //Print currentTarget vector values
        qInfo() << "TARGET:" << currentTarget.x() << currentTarget.y() << currentTarget.z();

        currentTarget.normalize();

        velocity[0] = currentTarget.x() * max_person_speed;
        velocity[1] = currentTarget.y() * max_person_speed;
        velocity[2] = 0.f;
        velocity[3] = 0.f;
        velocity[4] = 0.f;
        velocity[5] = 0.f;
        //Print velocity vector values
        qInfo() << "SPEED:" << velocity[0] << velocity[1] << velocity[2];
    }
    // SET VELOCITY MODULE IN PERSON NODE HARCODED FOR EVERY PERSON IN GRAPH!!!!!!!!!!!
    //Calculate velocity vector module
    float velocity_module = std::sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1]);

//    auto person_nodes = G->get_nodes_by_type("person");
//    // Iterate over person nodes
//    for (const auto& person_node : person_nodes)
//    {
//        G->add_or_modify_attrib_local<velocity_module_att>(person_node, velocity_module);
//    }

    humanNode->setVelocity(velocity);
}
void SpecificWorker::Webots2Robocomp_resetWebots()
{
    std::cout << "##################### RESET POSITION #####################" << std::endl;
    setElementsToStartPosition();
}
void SpecificWorker::Webots2Robocomp_setPathToHuman(int humanId, RoboCompGridder::TPath path)
{

    if(humanObjects[humanId].node == nullptr)
    {
        qInfo() << "Human not found";
        return;
    }

    if(humanObjects[humanId].path.empty() and path.size() > 3)
    {
        webots::Node *robotNode = robot->getFromDef("shadow");
        RoboCompGridder::TPath transformed_path;

        auto x = robotNode->getField("translation")->getSFVec3f()[0] * 1000;
        auto y = robotNode->getField("translation")->getSFVec3f()[1] * 1000;
        auto rot = robotNode->getField("rotation")->getSFRotation();

        Eigen::Vector3d axis(rot[0], rot[1], rot[2]);
        Eigen::AngleAxisd angleAxis(rot[3], axis.normalized());
        Eigen::Quaterniond quaternion(angleAxis);

        Eigen::Vector3d eulerAngles = quaternion.toRotationMatrix().eulerAngles(0, 1, 2);

//    //Get transform matrix

        auto tf = create_affine_matrix(eulerAngles.x(), eulerAngles.y(), eulerAngles.z(), Eigen::Vector3d {x, y, 0});

        //    //Print tf matrix values
//    qInfo() << "Transform matrix" << tf(0,0) << tf(0,1) << tf(0,2) << tf(0,3) << tf(1,0) << tf(1,1) << tf(1,2) << tf(1,3) << tf(2,0) << tf(2,1) << tf(2,2) << tf(2,3) << tf(3,0) << tf(3,1) << tf(3,2) << tf(3,3);

        for(auto &p : path)
        {
            Eigen::Vector2f tf_point = (tf.matrix() * Eigen::Vector4d(p.y , -p.x, 0.0, 1.0)).head(2).cast<float>();
            transformed_path.emplace_back(RoboCompGridder::TPoint(tf_point.x() / 1000, tf_point.y() / 1000 , 100.0));
            qInfo() << __FUNCTION__ << "point?" << tf_point.x() << tf_point.y();
        }

        humanObjects[humanId].path = transformed_path;
    }

}

#pragma endregion Webots2Robocomp Methods

void SpecificWorker::printNotImplementedWarningMessage(string functionName)
{
    cout << "Function not implemented used: " << "[" << functionName << "]" << std::endl;
}


Matrix4d SpecificWorker::create_affine_matrix(double a, double b, double c, Vector3d trans)
{
    Transform<double, 3, Eigen::Affine> t;
    t = Translation<double, 3>(trans);
    t.rotate(AngleAxis<double>(a, Vector3d::UnitX()));
    t.rotate(AngleAxis<double>(b, Vector3d::UnitY()));
    t.rotate(AngleAxis<double>(c, Vector3d::UnitZ()));
    return t.matrix();
}

void SpecificWorker::setObstacleStartPosition()
{
    webots::Node *obstacleNode = robot->getFromDef("OBSTACLE");
    obstacleStartPosition[0] = obstacleNode->getField("translation")->getSFVec3f()[0];
    obstacleStartPosition[1] = obstacleNode->getField("translation")->getSFVec3f()[1];
    obstacleStartPosition[2] = obstacleNode->getField("translation")->getSFVec3f()[2];
}

void SpecificWorker::setElementsToStartPosition()
{

    //Increase experiment Counter
    experiment_id++;
    // Set the obstacle to the initial position
    webots::Node *obstacleNode = robot->getFromDef("OBSTACLE");
    // Get point in a defined radius respect the initial position
//    auto random_point = getRandomPointInRadius(obstacleStartPosition[0], obstacleStartPosition[1], 0.5);
    auto random_point_obs = getRandomPointInLine(obstaclePoseLine);
    const double obstaclePosition[] = {random_point_obs.x, random_point_obs.y, 0.28};
    obstacleNode->getField("translation")->setSFVec3f(obstaclePosition);
    // Print obstacle position
    std::cout << "Obstacle position: " << obstaclePosition[0] << " " << obstaclePosition[1] << " " << obstaclePosition[2] << std::endl;

    // Set robot to start position and orientation
    auto random_point_robot = getRandomPointInLine(robotPoseLine1);
    const double robotPosition[] = {random_point_robot.x, random_point_robot.y, 0.0};
    robot->getFromDef("shadow")->getField("translation")->setSFVec3f(robotPosition);
    const double robotOrientation[] = {robotStartOrientation[0], robotStartOrientation[1], robotStartOrientation[2], robotStartOrientation[3]};
    robot->getFromDef("shadow")->getField("rotation")->setSFRotation(robotOrientation);
    // Set humans to start position
    for (auto &human : humanObjects)
    {
        webots::Node *humanNode = human.second.node;
        auto random_point_person = getRandomPointInLine(personPoseLine);
//        auto random_point = getRandomPointInRadius(human.second.startPosition[0], human.second.startPosition[1], 0.3);
        const double humanPosition[] = {random_point_person.x, random_point_person.y, 0.0};
        humanNode->getField("translation")->setSFVec3f(humanPosition);
        // Print human position
        std::cout << "Human position: " << humanPosition[0] << " " << humanPosition[1] << " " << humanPosition[2] << std::endl;
    }
}
void SpecificWorker::reset_sim()
{
    //Get and delete all nodes type object
    auto nodes = G->get_nodes_by_type("object");
    for (auto node : nodes)
    {
        G->delete_node(node);
        //Print deleted node
        qInfo() << __FUNCTION__ << "Deleted node: " << node.id();
    }
    //Get and delete all person nodes
    auto person_nodes = G->get_nodes_by_type("person");
    for (auto node : person_nodes)
    {
        G->delete_node(node);
        //Print deleted node
        qInfo() << __FUNCTION__ << "Deleted node: " << node.id();
    }
    auto intention_nodes = G->get_nodes_by_type("intention");
    for (auto node : intention_nodes)
    {
        G->delete_node(node);
        //Print deleted node
        qInfo() << __FUNCTION__ << "Deleted node: " << node.id();
    }
}
std::pair<double, double> SpecificWorker::getRandomPointInRadius(double centerX, double centerY, double radius) {
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate random angle
    std::uniform_real_distribution<double> angleDist(0.0, 2.0 * M_PI);
    double angle = angleDist(gen);

    // Generate random distance from center
    std::uniform_real_distribution<double> distanceDist(0.0, radius);
    double distance = distanceDist(gen);

    // Calculate coordinates of the random point
    double x = centerX + distance * cos(angle);
    double y = centerY + distance * sin(angle);

    return std::make_pair(x, y);
}
// Function to get a random point in a line between two points
RoboCompGridder::TPoint SpecificWorker::getRandomPointInLine(PoseLine pl)
{
    // Seed the random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Generate random distance from the first point
    std::uniform_real_distribution<double> distanceDist(0.0, 1.0);
    double distance = distanceDist(gen);

    // Calculate coordinates of the random point
    double x = pl.p1.x + distance * (pl.p2.x - pl.p1.x);
    double y = pl.p1.y + distance * (pl.p2.y - pl.p1.y);

    return RoboCompGridder::TPoint{.x=x, .y=y};
}
std::tuple<int, int, Eigen::Vector2f> SpecificWorker::calculate_collision_metrics()
{
    int possible_collision_exists = 0;
    int collision_solver_intention_exists = 0;
    Eigen::Vector2f collision_solver_pose = {0, 0};

    // Check if collision edges exists
    auto collision_edges = G->get_edges_by_type("collision");
    if(!collision_edges.empty())
        possible_collision_exists = 1;

    // Check if intention to solve the collision exists
    auto intention_nodes = G->get_nodes_by_type("intention");
    // Check if any intention node with "avoid_collision" name exists
    for(const auto &intention_node : intention_nodes)
        if(intention_node.name() == "avoid_collision")
        {
            collision_solver_intention_exists = 1;
            if(auto target_x = G->get_attrib_by_name<robot_target_x_att>(intention_node); target_x.has_value())
                collision_solver_pose.x() = target_x.value();
            if(auto target_y = G->get_attrib_by_name<robot_target_y_att>(intention_node); target_y.has_value())
                collision_solver_pose.y() = target_y.value();
        }
    return std::make_tuple(possible_collision_exists, collision_solver_intention_exists, collision_solver_pose);
}

std::vector<std::vector<double>> SpecificWorker::getPositions()
{
    std::vector<std::vector<double>> positions;

    auto shadow_node = robot->getFromDef("shadow");
    auto target_node = robot->getFromDef("SOFA");
    auto obstacle_node = robot->getFromDef("OBSTACLE");
    auto human_node = robot->getFromDef("HUMAN_1");

    if(shadow_node == nullptr){ cerr << "Robot shadow does not exists."; return {}; }
    if(target_node == nullptr){ cerr << "Couch does not exists."; return {}; }
    if(human_node == nullptr){ cerr << "Human 0 does not exists."; return {}; }
    if(obstacle_node == nullptr){ cerr << "Soccer ball does not exists."; return {}; }

    positions.push_back({human_node->getField("translation")->getSFVec3f()[0], human_node->getField("translation")->getSFVec3f()[1]});
    positions.push_back({obstacle_node->getField("translation")->getSFVec3f()[0], obstacle_node->getField("translation")->getSFVec3f()[1]});
    positions.push_back({target_node->getField("translation")->getSFVec3f()[0], target_node->getField("translation")->getSFVec3f()[1]});
    positions.push_back({shadow_node->getField("translation")->getSFVec3f()[0], shadow_node->getField("translation")->getSFVec3f()[1], shadow_node->getField("rotation")->getSFRotation()[0]});

    return positions;
}
void SpecificWorker::modify_node_slot(std::uint64_t id, const std::string &type)
{
    if(type == "intention")
        if(auto intention_node = G->get_node(id); intention_node.has_value())
            if(intention_node.value().name() == "STOP")
                reset = true;
}
/**************************************/
// From the RoboCompCamera360RGB you can use this types:
// RoboCompCamera360RGB::TRoi
// RoboCompCamera360RGB::TImage

/**************************************/
// From the RoboCompCameraRGBDSimple you can use this types:
// RoboCompCameraRGBDSimple::Point3D
// RoboCompCameraRGBDSimple::TPoints
// RoboCompCameraRGBDSimple::TImage
// RoboCompCameraRGBDSimple::TDepth
// RoboCompCameraRGBDSimple::TRGBD

/**************************************/
// From the RoboCompLaser you can use this types:
// RoboCompLaser::LaserConfData
// RoboCompLaser::TData

/**************************************/
// From the RoboCompLidar3D you can use this types:
// RoboCompLidar3D::TPoint
// RoboCompLidar3D::TDataImage
// RoboCompLidar3D::TData

/**************************************/
// From the RoboCompOmniRobot you can use this types:
// RoboCompOmniRobot::TMechParams

/**************************************/
// From the RoboCompVisualElements you can use this types:
// RoboCompVisualElements::TRoi
// RoboCompVisualElements::TObject
// RoboCompVisualElements::TObjects

/**************************************/
// From the RoboCompWebots2Robocomp you can use this types:
// RoboCompWebots2Robocomp::Vector3
// RoboCompWebots2Robocomp::Quaternion

/**************************************/
// From the RoboCompJoystickAdapter you can use this types:
// RoboCompJoystickAdapter::AxisParams
// RoboCompJoystickAdapter::ButtonParams
// RoboCompJoystickAdapter::TData

