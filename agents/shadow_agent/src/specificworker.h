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

/**
	\brief
	@author authorname
*/



#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H


// If you want to reduce the period automatically due to lack of use, you must uncomment the following line
//#define HIBERNATION_ENABLED

#include <genericworker.h>
#include <doublebuffer/DoubleBuffer.h>

/**
 * \brief Class SpecificWorker implements the core functionality of the component.
 */
class SpecificWorker : public GenericWorker
{
	Q_OBJECT
	public:
	    /**
	     * \brief Constructor for SpecificWorker.
	     * \param configLoader Configuration loader for the component.
	     * \param tprx Tuple of proxies required for the component.
	     * \param startup_check Indicates whether to perform startup checks.
	     */
		SpecificWorker(const ConfigLoader& configLoader, TuplePrx tprx, bool startup_check);

		/**
	     * \brief Destructor for SpecificWorker.
	     */
		~SpecificWorker();



		void FullPoseEstimationPub_newFullPose(RoboCompFullPoseEstimation::FullPoseEuler pose);

	public slots:

		/**
		 * \brief Initializes the worker one time.
		 */
		void initialize();

		/**
		 * \brief Main compute loop of the worker.
		 */
		void compute();

		/**
		 * \brief Handles the emergency state loop.
		 */
		void emergency();

		/**
		 * \brief Restores the component from an emergency state.
		 */
		void restore();

	    /**
	     * \brief Performs startup checks for the component.
	     * \return An integer representing the result of the checks.
	     */
		int startup_check();

		void modify_node_slot(std::uint64_t, const std::string &type){};
		void modify_node_attrs_slot(std::uint64_t id, const std::vector<std::string>& att_names){};
		void modify_edge_slot(std::uint64_t from, std::uint64_t to,  const std::string &type){};
		void modify_edge_attrs_slot(std::uint64_t from, std::uint64_t to, const std::string &type, const std::vector<std::string>& att_names){};
		void del_edge_slot(std::uint64_t from, std::uint64_t to, const std::string &edge_tag){};
		void del_node_slot(std::uint64_t from){};

	private:
		/**
	     * \brief DoubleBuffer for movind data between threads.
	     */
		DoubleBuffer<RoboCompFullPoseEstimation::FullPoseEuler, RoboCompFullPoseEstimation::FullPoseEuler> pose_buffer;
		/**
	     * \brief Flag indicating whether startup checks are enabled.
	     */
		bool startup_check_flag;

	signals:
		//void customSignal();
};

#endif
