/*
 *    Copyright (C) 2024 by YOUR NAME HERE
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
#include "doublebuffer/DoubleBuffer.h"
#include <Eigen/Eigen>
#include "abstract_graphic_viewer/abstract_graphic_viewer.h"
#include <fps/fps.h>
#include <person.h>

class SpecificWorker : public GenericWorker
{
    Q_OBJECT
    public:
        SpecificWorker(TuplePrx tprx, bool startup_check);
        ~SpecificWorker();
        bool setParams(RoboCompCommonBehavior::ParameterList params);
        void VisualElementsPub_setVisualObjects(RoboCompVisualElementsPub::TData data);

    public slots:
        void compute();
        int startup_check();
        void initialize(int period);

    private:
        bool startup_check_flag;

        //Graphics
        AbstractGraphicViewer *viewer;

        // Lidar Thread
        DoubleBuffer<std::vector<Eigen::Vector3f>, std::vector<Eigen::Vector3f>> buffer_lidar_data;
        std::thread read_lidar_th;
        void read_lidar();

        // Pilar cone parameters
        float cone_radius = 3000;
        float cone_angle = 1;   // rads

        struct Params
        {
            float ROBOT_WIDTH = 460;  // mm
            float ROBOT_LENGTH = 480;  // mm
            float ROBOT_SEMI_WIDTH = ROBOT_WIDTH / 2.f;     // mm
            float ROBOT_SEMI_LENGTH = ROBOT_LENGTH / 2.f;    // mm
            float TILE_SIZE = 100;   // mm
            float MIN_DISTANCE_TO_TARGET = ROBOT_WIDTH / 2.f; // mm
            std::string LIDAR_NAME_LOW = "bpearl";
            std::string LIDAR_NAME_HIGH = "helios";
            float MAX_LIDAR_LOW_RANGE = 10000;  // mm
            float MAX_LIDAR_HIGH_RANGE = 10000;  // mm
            float MAX_LIDAR_RANGE = 10000;  // mm used in the grid
            int LIDAR_LOW_DECIMATION_FACTOR = 2;
            int LIDAR_HIGH_DECIMATION_FACTOR = 1;
            QRectF GRID_MAX_DIM{-6000, -6000, 12000, 12000};
            float CARROT_DISTANCE = 400;   // mm
            float CARROT_ANGLE = M_PI_4 / 6.f;   // rad
            long PERIOD_HYSTERESIS = 2; // to avoid oscillations in the adjustment of the lidar thread period
            int PERIOD = 100;    // ms (10 Hz) for compute timer
            float MIN_ANGLE_TO_TARGET = 1.f;   // rad
            int MPC_HORIZON = 8;
            bool USE_MPC = true;
            unsigned int ELAPSED_TIME_BETWEEN_PATH_UPDATES = 3000;
            int NUM_PATHS_TO_SEARCH = 3;
            float MIN_DISTANCE_BETWEEN_PATHS = 500; // mm
            unsigned int SECS_TO_GET_IN = 1; // secs
            unsigned int SECS_TO_GET_OUT = 2; // sec//

            // YOLO
            int STOP_SIGN = 11;
            int PERSON = 0;
            int BENCH = 13;
            int CHAIR = 56;

            // colors
            QColor TARGET_COLOR= {"orange"};
            QColor LIDAR_COLOR = {"LightBlue"};
            QColor PATH_COLOR = {"orange"};
            QColor SMOOTHED_PATH_COLOR = {"magenta"};
        };
        Params params;

        std::vector<std::string> YOLO_NAMES = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
        "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
        "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
        "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"};

    struct Object
        {
            Object() = default;
            RoboCompVisualElementsPub::TObject obj;
            QGraphicsRectItem *item = nullptr;
            bool is_target = false;
            std::chrono::high_resolution_clock::time_point insertion_time, last_update_time;

            int get_id() const {return obj.id;}
            QGraphicsItem *get_item() const {return item;}
            void init_item(QGraphicsScene *scene, float x, float y, float width, float height)
            {
                item = scene->addRect(-width / 2.f, -height / 2.f, width, height,
                                      QPen(QColor("magenta")), QBrush(QColor("magenta")));
                item->setPos(x, y);
                // add a text item with the id
                auto text = scene->addText(QString::number(obj.id));
                text->setParentItem(item);
                text->setPos(-text->boundingRect().width() * 5, text->boundingRect().height() * 17);
                text->setDefaultTextColor(QColor("black"));
                text->setScale(10);
                QTransform transform; transform.scale(1, -1);
                text->setTransform(transform);
            }
            void update_last_update_time() { last_update_time = std::chrono::high_resolution_clock::now(); };
            void set_insertion_time() { insertion_time = std::chrono::high_resolution_clock::now(); update_last_update_time();}
            std::chrono::high_resolution_clock::time_point get_insertion_time() const {return insertion_time;};
            std::chrono::high_resolution_clock::time_point get_last_update_time() const {return last_update_time;};
            void update_attributes(const RoboCompVisualElementsPub::TObject &object){};
            void set_object_data(const RoboCompVisualElementsPub::TObject &object){ obj = object;};
            void remove_item(QGraphicsScene *scene) { scene->removeItem(item); delete item; item = nullptr;};
        };

        // People
        Person wanted_person;
        using People = std::vector<Person>;
        using Objects = std::vector<Object>;
        People people;
        Objects objects;

        // Robot path
        std::vector<QGraphicsEllipseItem*> points;

        // Visual elements
        DoubleBuffer<RoboCompVisualElementsPub::TData, RoboCompVisualElementsPub::TData> buffer_visual_elements;
        DoubleBuffer<RoboCompVisualElementsPub::TData, RoboCompVisualElementsPub::TData> buffer_room_elements;
        void draw_lidar(const vector<Eigen::Vector3f> &points, int decimate);
        void draw_room(const RoboCompVisualElementsPub::TObject &obj);
        void draw_path(const std::vector<Eigen::Vector2f> &path, QGraphicsScene *scene, bool erase_only=false);
        void process_people(const RoboCompVisualElementsPub::TData &data);
        void process_room(const RoboCompVisualElementsPub::TData &data);
        void print_people(const People &ppol);

        // fps
        FPSCounter fps;
        int hz = 0;

        void process_room_objects(const RoboCompVisualElementsPub::TData &data);
        void postprocess_target_person(const People &people_);
};

#endif
