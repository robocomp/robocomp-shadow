#ifndef ROOM_VISUALIZER_3D_H
#define ROOM_VISUALIZER_3D_H

#include <Qt3DCore/QEntity>
#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DExtras/QSphereMesh>
#include <Qt3DExtras/QCuboidMesh>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QCamera>
#include <Qt3DRender/QMesh>
#include <Qt3DRender/QPointLight>
#include <QVector3D>
#include <QColor>
#include <QDebug>
#include <Eigen/Dense>
#include <vector>
#include <Lidar3D.h>
#include <QWidget>
#include "custom_camera_controller.h"

class RoomVisualizer3D : public QObject
{
    Q_OBJECT

public:
    RoomVisualizer3D(const QString &robotMeshPath = "src/meshes/shadow.obj");

    ~RoomVisualizer3D();

    // Main update methods
    void updatePointCloud(const std::vector<Eigen::Vector3f> &points);

    void updatePointCloud(const RoboCompLidar3D::TPoints &points);

    void updateRoom(float half_width, float half_depth);

    void updateRobotPose(const Eigen::Vector3f &robot_pose);

    void draw_door(float x, float y, float z, float theta, float width, float height, float open_angle);

    // Optional: uncertainty visualization
    void updateUncertainty(float pos_std_x, float pos_std_y, float theta_std);

    // Show/hide elements
    void showPointCloud(bool visible);

    void showRoom(bool visible);

    void showRobot(bool visible);

    void showUncertainty(bool visible);

    // Camera control
    void resetCamera();

    void setCameraDistance(float distance);

    // Window control
    void show();

    void hide();

    Qt3DExtras::Qt3DWindow *getWindow() { return view; }
    QWidget *getWidget() { return QWidget::createWindowContainer(view); }

    // Debug/testing
    void setRobotVisibility(bool visible);

    void forceRobotBox(); // Force use of box instead of mesh for testing

    // Visual settings
    void setRoomLineWidth(float width); // Set room wireframe line thickness (in meters)
    void setUncertaintyScale(float scale);

    // Amplify uncertainty visualization (default 1.0, try 5.0-10.0 for small uncertainties)

private:
    void setupScene();

    void setupCamera();

    void setupLighting();

    void createCoordinateAxes();

    void createGroundGrid(); // Changed from createGroundPlane
    Qt3DCore::QEntity *createLineBox(float width, float height, const QColor &color);

    Qt3DCore::QEntity *createUncertaintyEllipse(float std_x, float std_y, const QColor &color);

    // Qt3D components
    Qt3DExtras::Qt3DWindow *view;
    Qt3DCore::QEntity *rootEntity = nullptr;
    Qt3DCore::QEntity *sceneEntity;
    Qt3DRender::QCamera *camera;
    CustomCameraController *camController;

    // Visualization entities
    QVector<Qt3DCore::QEntity *> pointEntities;
    Qt3DCore::QEntity *roomEntity;
    Qt3DCore::QEntity *robotEntity;
    Qt3DCore::QEntity *uncertaintyEntity;
    Qt3DCore::QEntity *axesEntity;
    Qt3DCore::QEntity *groundEntity;
    Qt3DCore::QEntity *doorEntity = nullptr;

    // Shared resources (for instancing efficiency)
    Qt3DExtras::QSphereMesh *sharedPointMesh;
    Qt3DExtras::QPhongMaterial *sharedPointMaterial;
    Qt3DRender::QMesh *robotMesh;
    Qt3DExtras::QCuboidMesh *fallbackRobotMesh; // Fallback if .obj doesn't load
    Qt3DCore::QTransform *robotTransform;
    Qt3DCore::QTransform *roomTransform; // Transform for room scaling
    Qt3DCore::QTransform *uncertaintyTransform; // Transform for uncertainty scaling

    // Settings
    QString robotMeshPath;
    float pointRadius;
    float roomLineWidth; // Width of room wireframe lines
    float uncertaintyScale; // Scale factor for uncertainty visualization
    QColor pointColor;
    QColor roomColor;
    QColor robotColor;
    QColor uncertaintyColor;

    // door transforms
    Qt3DCore::QTransform *doorTransform_ = nullptr;
    Qt3DCore::QTransform *hingeTransform_ = nullptr;
};

#endif // ROOM_VISUALIZER_3D_H
