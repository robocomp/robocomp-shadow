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

/**
	\brief
	@author authorname
*/



#ifndef SPECIFICWORKER_H
#define SPECIFICWORKER_H

#include <genericworker.h>
#include "grid.h"
#include "doublebuffer/DoubleBuffer.h"
#include <Eigen/Eigen>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <fps/fps.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);

        void SegmentatorTrackingPub_setTrack(int track);

    public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        bool startup_check_flag;
        DoubleBuffer<Eigen::Vector2f,Eigen::Vector2f> target_buffer;

        //Graphics
        AbstractGraphicViewer *viewer;
        std::vector<Eigen::Vector3f> get_lidar_data();

        //GRID
        int z_lidar_height = 1250;
        Grid grid;

        // FPS
        FPSCounter fps;

        // Target
        struct Target
        {
            bool active = false;
            Eigen::Vector2f point;
            QPointF qpoint;
            void set(QPointF p)
            {
                point = Eigen::Vector2f(p.x(), p.y());
                qpoint = p;
                active = true;
            }
            void unset() { active = false; }
        };
        Target target;

        // Path
        void draw_path(const vector<Eigen::Vector2f> &path, QGraphicsScene *scene);

    RoboCompGridPlanner::TPoint send_path(const vector<Eigen::Vector2f> &path,
                                          float threshold_dist, float threshold_angle);
    void draw_subtarget(const Eigen::Vector2f &point, QGraphicsScene *scene);

};

#endif
