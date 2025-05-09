/*
 *    Copyright (C) 2025 by YOUR NAME HERE
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


/** \mainpage RoboComp::base_controller_agent
 *
 * \section intro_sec Introduction
 *
 * The base_controller_agent component...
 *
 * \section interface_sec Interface
 *
 * interface...
 *
 * \section install_sec Installation
 *
 * \subsection install1_ssec Software depencences
 * ...
 *
 * \subsection install2_ssec Compile and install
 * cd base_controller_agent
 * <br>
 * cmake . && make
 * <br>
 * To install:
 * <br>
 * sudo make install
 *
 * \section guide_sec User guide
 *
 * \subsection config_ssec Configuration file
 *
 * <p>
 * The configuration file etc/config...
 * </p>
 *
 * \subsection execution_ssec Execution
 *
 * Just: "${PATH_TO_BINARY}/base_controller_agent --Ice.Config=${PATH_TO_CONFIG_FILE}"
 *
 * \subsection running_ssec Once running
 *
 * ...
 *
 */
#include <signal.h>

// QT includes
#include <QtCore>
#include <QtWidgets>

// ICE includes
#include <Ice/Ice.h>
#include <IceStorm/IceStorm.h>
#include <Ice/Application.h>

#include <ConfigLoader/ConfigLoader.h>

#include <sigwatch/sigwatch.h>

#include "genericworker.h"
#include "../src/specificworker.h"

#include <fullposeestimationpubI.h>

#include <FullPoseEstimation.h>

#define USE_QTGUI

#define PROGRAM_NAME    "base_controller_agent"
#define SERVER_FULL_NAME   "RoboComp base_controller_agent::base_controller_agent"


class base_controller_agent : public Ice::Application
{
public:
	base_controller_agent (QString configFile, QString prfx, bool startup_check) { 
		this->configFile = configFile.toStdString();
		this->prefix = prfx.toStdString();
		this->startup_check_flag=startup_check; 

		this->configLoader.load(this->configFile);
		this->configLoader.printConfig();
		}

	Ice::InitializationData getInitializationDataIce();

private:
	void initialize();
	std::string prefix, configFile;
	ConfigLoader configLoader;
	TuplePrx tprx;
	bool startup_check_flag = false;

public:
	virtual int run(int, char*[]);
};

Ice::InitializationData base_controller_agent::getInitializationDataIce(){
        Ice::InitializationData initData;
        initData.properties = Ice::createProperties();
        initData.properties->setProperty("Ice.Warn.Connections", this->configLoader.get<std::string>("Ice.Warn.Connections"));
        initData.properties->setProperty("Ice.Trace.Network", this->configLoader.get<std::string>("Ice.Trace.Network"));
        initData.properties->setProperty("Ice.Trace.Protocol", this->configLoader.get<std::string>("Ice.Trace.Protocol"));
        initData.properties->setProperty("Ice.MessageSizeMax", this->configLoader.get<std::string>("Ice.MessageSizeMax"));
		return initData;
}

void base_controller_agent::initialize()
{
    this->configLoader.load(this->configFile);
	this->configLoader.printConfig();
}

int base_controller_agent::run(int argc, char* argv[])
{
#ifdef USE_QTGUI
	QApplication a(argc, argv);  // GUI application
#else
	QCoreApplication a(argc, argv);  // NON-GUI application
#endif


	sigset_t sigs;
	sigemptyset(&sigs);
	sigaddset(&sigs, SIGHUP);
	sigaddset(&sigs, SIGINT);
	sigaddset(&sigs, SIGTERM);
	sigprocmask(SIG_UNBLOCK, &sigs, 0);

	UnixSignalWatcher sigwatch;
	sigwatch.watchForSignal(SIGINT);
	sigwatch.watchForSignal(SIGTERM);
	QObject::connect(&sigwatch, SIGNAL(unixSignal(int)), &a, SLOT(quit()));

	int status=EXIT_SUCCESS;

	RoboCompGridPlanner::GridPlannerPrxPtr gridplanner_proxy;
	RoboCompLidar3D::Lidar3DPrxPtr lidar3d_proxy;

	std::string proxy, tmp;
	initialize();

	try
	{
	    proxy = configLoader.get<std::string>("Proxies.GridPlanner");
		gridplanner_proxy = Ice::uncheckedCast<RoboCompGridPlanner::GridPlannerPrx>(communicator()->stringToProxy(proxy));
	}
	catch(const Ice::Exception& ex)
	{
		std::cout << "[" << PROGRAM_NAME << "]: Exception creating proxy GridPlanner: " << ex;
		return EXIT_FAILURE;
	}
	qInfo("GridPlannerProxy initialized Ok!");


	try
	{
	    proxy = configLoader.get<std::string>("Proxies.Lidar3D");
		lidar3d_proxy = Ice::uncheckedCast<RoboCompLidar3D::Lidar3DPrx>(communicator()->stringToProxy(proxy));
	}
	catch(const Ice::Exception& ex)
	{
		std::cout << "[" << PROGRAM_NAME << "]: Exception creating proxy Lidar3D: " << ex;
		return EXIT_FAILURE;
	}
	qInfo("Lidar3DProxy initialized Ok!");


	IceStorm::TopicManagerPrxPtr topicManager;
	try
	{
		topicManager = Ice::checkedCast<IceStorm::TopicManagerPrx>(communicator()->stringToProxy(configLoader.get<std::string>("Proxies.TopicManager")));
		if (!topicManager)
		{
		    std::cout << "[" << PROGRAM_NAME << "]: TopicManager.Proxy not defined in config file."<<std::endl;
		    std::cout << "	 Config line example: TopicManager.Proxy=IceStorm/TopicManager:default -p 9999"<<std::endl;
	        return EXIT_FAILURE;
		}
	}
	catch (const Ice::Exception &ex)
	{
		std::cout << "[" << PROGRAM_NAME << "]: Exception: 'rcnode' not running: " << ex << std::endl;
		return EXIT_FAILURE;
	}

	tprx = std::make_tuple(gridplanner_proxy,lidar3d_proxy);
	SpecificWorker *worker = new SpecificWorker(this->configLoader, tprx, startup_check_flag);
	QObject::connect(worker, SIGNAL(kill()), &a, SLOT(quit()));

	try
	{

		// Server adapter creation and publication
		std::shared_ptr<IceStorm::TopicPrx> fullposeestimationpub_topic;
		Ice::ObjectPrxPtr fullposeestimationpub;
		try
		{

		    tmp = configLoader.get<std::string>("Endpoints.FullPoseEstimationPubTopic");
			Ice::ObjectAdapterPtr FullPoseEstimationPub_adapter = communicator()->createObjectAdapterWithEndpoints("fullposeestimationpub", tmp);
			RoboCompFullPoseEstimationPub::FullPoseEstimationPubPtr fullposeestimationpubI_ =  std::make_shared <FullPoseEstimationPubI>(worker);
			auto fullposeestimationpub = FullPoseEstimationPub_adapter->addWithUUID(fullposeestimationpubI_)->ice_oneway();
			if(!fullposeestimationpub_topic)
			{
				try {
					fullposeestimationpub_topic = topicManager->create("FullPoseEstimationPub");
				}
				catch (const IceStorm::TopicExists&) {
					//Another client created the topic
					try{
						std::cout << "[" << PROGRAM_NAME << "]: Probably other client already opened the topic. Trying to connect.\n";
						fullposeestimationpub_topic = topicManager->retrieve("FullPoseEstimationPub");
					}
					catch(const IceStorm::NoSuchTopic&)
					{
						std::cout << "[" << PROGRAM_NAME << "]: Topic doesn't exists and couldn't be created.\n";
						//Error. Topic does not exist
					}
				}
				catch(const IceUtil::NullHandleException&)
				{
					std::cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
					"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
					return EXIT_FAILURE;
				}
				IceStorm::QoS qos;
				fullposeestimationpub_topic->subscribeAndGetPublisher(qos, fullposeestimationpub);
			}
			FullPoseEstimationPub_adapter->activate();
		}
		catch(const IceStorm::NoSuchTopic&)
		{
			std::cout << "[" << PROGRAM_NAME << "]: Error creating FullPoseEstimationPub topic.\n";
			//Error. Topic does not exist
		}


		// Server adapter creation and publication
		std::cout << SERVER_FULL_NAME " started" << std::endl;

		// User defined QtGui elements ( main window, dialogs, etc )

		#ifdef USE_QTGUI
			//ignoreInterrupt(); // Uncomment if you want the component to ignore console SIGINT signal (ctrl+c).
			a.setQuitOnLastWindowClosed( true );
		#endif
		// Run QT Application Event Loop
		a.exec();

		try
		{
			std::cout << "Unsubscribing topic: fullposeestimationpub " <<std::endl;
			fullposeestimationpub_topic->unsubscribe( fullposeestimationpub );
		}
		catch(const Ice::Exception& ex)
		{
			std::cout << "ERROR Unsubscribing topic: fullposeestimationpub " << ex.what()<<std::endl;
		}


		status = EXIT_SUCCESS;
	}
	catch(const Ice::Exception& ex)
	{
		status = EXIT_FAILURE;

		std::cerr << "[" << PROGRAM_NAME << "]: Exception raised on main thread: " << std::endl;
		std::cerr << ex;

	}
	#ifdef USE_QTGUI
		a.quit();
	#endif

	status = EXIT_SUCCESS;
	delete worker;
	return status;
}

int main(int argc, char* argv[])
{
	std::string arg;

	// Set config file
	QString configFile("etc/config");
	bool startup_check_flag = false;
	QString prefix("");
	if (argc > 1)
	{

		// Search in argument list for arguments
		QString startup = QString("--startup-check");
		QString initIC = QString("--Ice.Config=");
		QString prfx = QString("--prefix=");
		for (int i = 0; i < argc; ++i)
		{
			arg = argv[i];
			if (arg.find(startup.toStdString(), 0) != std::string::npos)
			{
				startup_check_flag = true;
				std::cout << "Startup check = True"<< std::endl;
			}
			else if (arg.find(prfx.toStdString(), 0) != std::string::npos)
			{
				prefix = QString::fromStdString(arg).remove(0, prfx.size());
				if (prefix.size()>0)
					prefix += QString(".");
				printf("Configuration prefix: <%s>\n", prefix.toStdString().c_str());
			}
			else if (arg.find(initIC.toStdString(), 0) != std::string::npos)
			{
				configFile = QString::fromStdString(arg).remove(0, initIC.size());
				qDebug()<<__LINE__<<"Starting with config file:"<<configFile;
			}
			else if (i==1 and argc==2 and arg.find("--", 0) == std::string::npos)
			{
				configFile = QString::fromStdString(arg);
				qDebug()<<__LINE__<<QString::fromStdString(arg)<<argc<<arg.find("--", 0)<<"Starting with config file:"<<configFile;
			}
		}

	}
	base_controller_agent app(configFile, prefix, startup_check_flag);

	return app.main(argc, argv, app.getInitializationDataIce());
}
