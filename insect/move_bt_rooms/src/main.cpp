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


/** \mainpage RoboComp::move_bt_rooms
 *
 * \section intro_sec Introduction
 *
 * The move_bt_rooms component...
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
 * cd move_bt_rooms
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
 * Just: "${PATH_TO_BINARY}/move_bt_rooms --Ice.Config=${PATH_TO_CONFIG_FILE}"
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

#include <rapplication/rapplication.h>
#include <sigwatch/sigwatch.h>
#include <qlog/qlog.h>

#include "config.h"
#include "genericmonitor.h"
#include "genericworker.h"
#include "specificworker.h"
#include "specificmonitor.h"
#include "commonbehaviorI.h"

#include <omnirobotI.h>

#include <GenericBase.h>



class move_bt_rooms : public RoboComp::Application
{
public:
	move_bt_rooms (QString prfx, bool startup_check) { prefix = prfx.toStdString(); this->startup_check_flag=startup_check; }
private:
	void initialize();
	std::string prefix;
	TuplePrx tprx;
	bool startup_check_flag = false;

public:
	virtual int run(int, char*[]);
};

void ::move_bt_rooms::initialize()
{
	// Config file properties read example
	// configGetString( PROPERTY_NAME_1, property1_holder, PROPERTY_1_DEFAULT_VALUE );
	// configGetInt( PROPERTY_NAME_2, property1_holder, PROPERTY_2_DEFAULT_VALUE );
}

int ::move_bt_rooms::run(int argc, char* argv[])
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

	RoboCompGridPlanner::GridPlannerPrxPtr gridplanner_pubproxy;
	RoboCompGridPlanner::GridPlannerPrxPtr gridplanner_proxy;
	RoboCompLidar3D::Lidar3DPrxPtr lidar3d_proxy;
	RoboCompOmniRobot::OmniRobotPrxPtr omnirobot_proxy;

	string proxy, tmp;
	initialize();

	try
	{
		if (not GenericMonitor::configGetString(communicator(), prefix, "GridPlannerProxy", proxy, ""))
		{
			cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy GridPlannerProxy\n";
		}
		gridplanner_proxy = Ice::uncheckedCast<RoboCompGridPlanner::GridPlannerPrx>( communicator()->stringToProxy( proxy ) );
	}
	catch(const Ice::Exception& ex)
	{
		cout << "[" << PROGRAM_NAME << "]: Exception creating proxy GridPlanner: " << ex;
		return EXIT_FAILURE;
	}
	rInfo("GridPlannerProxy initialized Ok!");


	try
	{
		if (not GenericMonitor::configGetString(communicator(), prefix, "Lidar3DProxy", proxy, ""))
		{
			cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy Lidar3DProxy\n";
		}
		lidar3d_proxy = Ice::uncheckedCast<RoboCompLidar3D::Lidar3DPrx>( communicator()->stringToProxy( proxy ) );
	}
	catch(const Ice::Exception& ex)
	{
		cout << "[" << PROGRAM_NAME << "]: Exception creating proxy Lidar3D: " << ex;
		return EXIT_FAILURE;
	}
	rInfo("Lidar3DProxy initialized Ok!");


	try
	{
		if (not GenericMonitor::configGetString(communicator(), prefix, "OmniRobotProxy", proxy, ""))
		{
			cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy OmniRobotProxy\n";
		}
		omnirobot_proxy = Ice::uncheckedCast<RoboCompOmniRobot::OmniRobotPrx>( communicator()->stringToProxy( proxy ) );
	}
	catch(const Ice::Exception& ex)
	{
		cout << "[" << PROGRAM_NAME << "]: Exception creating proxy OmniRobot: " << ex;
		return EXIT_FAILURE;
	}
	rInfo("OmniRobotProxy initialized Ok!");


	IceStorm::TopicManagerPrxPtr topicManager;
	try
	{
		topicManager = topicManager = Ice::checkedCast<IceStorm::TopicManagerPrx>(communicator()->propertyToProxy("TopicManager.Proxy"));
		if (!topicManager)
		{
		    cout << "[" << PROGRAM_NAME << "]: TopicManager.Proxy not defined in config file."<<endl;
		    cout << "	 Config line example: TopicManager.Proxy=IceStorm/TopicManager:default -p 9999"<<endl;
	        return EXIT_FAILURE;
		}
	}
	catch (const Ice::Exception &ex)
	{
		cout << "[" << PROGRAM_NAME << "]: Exception: 'rcnode' not running: " << ex << endl;
		return EXIT_FAILURE;
	}
	std::shared_ptr<IceStorm::TopicPrx> gridplanner_topic;

	while (!gridplanner_topic)
	{
		try
		{
			gridplanner_topic = topicManager->retrieve("GridPlanner");
		}
		catch (const IceStorm::NoSuchTopic&)
		{
			cout << "[" << PROGRAM_NAME << "]: ERROR retrieving GridPlanner topic. \n";
			try
			{
				gridplanner_topic = topicManager->create("GridPlanner");
			}
			catch (const IceStorm::TopicExists&){
				// Another client created the topic.
				cout << "[" << PROGRAM_NAME << "]: ERROR publishing the GridPlanner topic. It's possible that other component have created\n";
			}
		}
		catch(const IceUtil::NullHandleException&)
		{
			cout << "[" << PROGRAM_NAME << "]: ERROR TopicManager is Null. Check that your configuration file contains an entry like:\n"<<
			"\t\tTopicManager.Proxy=IceStorm/TopicManager:default -p <port>\n";
			return EXIT_FAILURE;
		}
	}

	auto gridplanner_pub = gridplanner_topic->getPublisher()->ice_oneway();
	gridplanner_pubproxy = Ice::uncheckedCast<RoboCompGridPlanner::GridPlannerPrx>(gridplanner_pub);

	tprx = std::make_tuple(gridplanner_proxy,lidar3d_proxy,omnirobot_proxy,gridplanner_pubproxy);
	SpecificWorker *worker = new SpecificWorker(tprx, startup_check_flag);
	//Monitor thread
	SpecificMonitor *monitor = new SpecificMonitor(worker,communicator());
	QObject::connect(monitor, SIGNAL(kill()), &a, SLOT(quit()));
	QObject::connect(worker, SIGNAL(kill()), &a, SLOT(quit()));
	monitor->start();

	if ( !monitor->isRunning() )
		return status;

	while (!monitor->ready)
	{
		usleep(10000);
	}

	try
	{
		try {
			// Server adapter creation and publication
			if (not GenericMonitor::configGetString(communicator(), prefix, "CommonBehavior.Endpoints", tmp, "")) {
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy CommonBehavior\n";
			}
			Ice::ObjectAdapterPtr adapterCommonBehavior = communicator()->createObjectAdapterWithEndpoints("commonbehavior", tmp);
			auto commonbehaviorI = std::make_shared<CommonBehaviorI>(monitor);
			adapterCommonBehavior->add(commonbehaviorI, Ice::stringToIdentity("commonbehavior"));
			adapterCommonBehavior->activate();
		}
		catch(const Ice::Exception& ex)
		{
			status = EXIT_FAILURE;

			cout << "[" << PROGRAM_NAME << "]: Exception raised while creating CommonBehavior adapter: " << endl;
			cout << ex;

		}



		try
		{
			// Server adapter creation and publication
			if (not GenericMonitor::configGetString(communicator(), prefix, "OmniRobot.Endpoints", tmp, ""))
			{
				cout << "[" << PROGRAM_NAME << "]: Can't read configuration for proxy OmniRobot";
			}
			Ice::ObjectAdapterPtr adapterOmniRobot = communicator()->createObjectAdapterWithEndpoints("OmniRobot", tmp);
			auto omnirobot = std::make_shared<OmniRobotI>(worker);
			adapterOmniRobot->add(omnirobot, Ice::stringToIdentity("omnirobot"));
			adapterOmniRobot->activate();
			cout << "[" << PROGRAM_NAME << "]: OmniRobot adapter created in port " << tmp << endl;
		}
		catch (const IceStorm::TopicExists&){
			cout << "[" << PROGRAM_NAME << "]: ERROR creating or activating adapter for OmniRobot\n";
		}


		// Server adapter creation and publication
		cout << SERVER_FULL_NAME " started" << endl;

		// User defined QtGui elements ( main window, dialogs, etc )

		#ifdef USE_QTGUI
			//ignoreInterrupt(); // Uncomment if you want the component to ignore console SIGINT signal (ctrl+c).
			a.setQuitOnLastWindowClosed( true );
		#endif
		// Run QT Application Event Loop
		a.exec();


		status = EXIT_SUCCESS;
	}
	catch(const Ice::Exception& ex)
	{
		status = EXIT_FAILURE;

		cout << "[" << PROGRAM_NAME << "]: Exception raised on main thread: " << endl;
		cout << ex;

	}
	#ifdef USE_QTGUI
		a.quit();
	#endif

	status = EXIT_SUCCESS;
	monitor->terminate();
	monitor->wait();
	delete worker;
	delete monitor;
	return status;
}

int main(int argc, char* argv[])
{
	string arg;

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
				cout << "Startup check = True"<< endl;
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
	::move_bt_rooms app(prefix, startup_check_flag);

	return app.main(argc, argv, configFile.toLocal8Bit().data());
}
