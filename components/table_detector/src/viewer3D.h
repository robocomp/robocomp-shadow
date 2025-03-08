//
// Created by pbustos on 25/02/25.
//

#ifndef VIEWER3D_H
#define VIEWER3D_H

#include <Qt3DCore/QEntity>
#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DCore/QTransform>
#include "protected_map.h"

namespace rc
{
    class Viewer3D final : public QWidget
    {
        Q_OBJECT
        public:
            explicit Viewer3D(QWidget *parent = nullptr, const std::shared_ptr<ActionablesData> &inner_model_ = nullptr);
            ~Viewer3D();

            public slots:
                void createRoomTransform();
                void createFridgeTransform();
                void updateFridgeTransform();
                void createTableTransform();
                void updateTableTransform();
                void updateRobotTransform(const Eigen::Affine2d &newTransform);

        private:
            std::shared_ptr<ActionablesData> inner_model;
            // The root entity of our 3D scene
            Qt3DCore::QEntity *m_rootEntity;
            // The Qt3D window embedded in this widget
            Qt3DExtras::Qt3DWindow *m_view;
            // The container widget for the Qt3DWindow
            QWidget *m_container;

            // Entities for the scene
            Qt3DCore::QEntity *m_floorEntity;
            Qt3DCore::QEntity *m_roomEntity;  // represents walls & floor of the room (top open)
            Qt3DCore::QEntity *m_robotEntity;
            Qt3DCore::QEntity *m_fridgeEntity;
            Qt3DCore::QEntity *m_tableEntity;

            // Transform for the fridge (to be updated via slot)
            Qt3DCore::QTransform *m_fridgeTransform;
            Qt3DCore::QTransform *m_robotTransform;
            Qt3DCore::QTransform *m_tableTransform;

            void add_lights();
            void add_floor();
            void add_robot();
            void add_axes();
            void add_scene_camera();
    };
};  // namespace rc
#endif //VIEWER3D_H
