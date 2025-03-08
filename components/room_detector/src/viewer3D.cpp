//
// Created by pbustos on 25/02/25.
//

#include "viewer3D.h"
#include <Qt3DExtras/Qt3DWindow>
#include <Qt3DExtras/QOrbitCameraController>
#include <Qt3DExtras/QPlaneMesh>
#include <Qt3DExtras/QCuboidMesh>
#include <Qt3DExtras/QPhongMaterial>
#include <Qt3DRender/QCamera>
#include <Qt3DCore/QTransform>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DRender/QMesh>
#include <QHBoxLayout>
#include <Qt3DRender/QPointLight> // Include point light
#include <Qt3DRender/QDirectionalLight> // Include directional light
#include <Qt3DRender/QShaderProgram> // Include shader program
#include <Qt3DRender/QTechnique> // Include technique
#include <Qt3DRender/QRenderPass> // Include render pass
#include <Qt3DRender/QEffect> // Include effect
#include <Qt3DRender/QMaterial> // Include material
#include <QPainter>
#include <QImage>
#include "/usr/include/x86_64-linux-gnu/qt6/Qt3DRender/QTexture"
#include <Qt3DExtras/QDiffuseMapMaterial>
#include <Qt3DExtras/QTextureMaterial>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QConeMesh>

namespace rc
{
    Viewer3D::Viewer3D(QWidget *parent, const std::shared_ptr<ActionablesData> &inner_model_)
        : QWidget(parent)
    {
        inner_model = inner_model_;
        // Root entity.
        m_rootEntity = new Qt3DCore::QEntity();

        // Create a Qt3DWindow and embed it in this widget.
        m_view = new Qt3DExtras::Qt3DWindow();
        m_view->defaultFrameGraph()->setClearColor(QColor(QRgb(0x4d4d4f)));
        m_container = QWidget::createWindowContainer(m_view, this);
        m_container->setMinimumSize(parent->width()*0.95, parent->height()*0.95);
        m_container->setFocusPolicy(Qt::TabFocus);
        auto *layout = new QHBoxLayout(this);
        layout->addWidget(m_container);
        setLayout(layout);

        // Set the root entity for the Qt3D window.
        m_view->setRootEntity(m_rootEntity);

        add_axes();
        add_floor();
        add_robot();
        add_scene_camera();
        add_lights();
    }

    Viewer3D::~Viewer3D()
    {
        // Save camera position
        QSettings settings("YourOrganizationName", "YourAppName");
        settings.setValue("cameraPosition", m_view->camera()->position());
        settings.setValue("cameraOrientation", m_view->camera()->transform()->rotation());
    }

    /////////////// SLOTS /////////////////////////////////////////////////////7
    void Viewer3D::updateRobotTransform(const Eigen::Affine2d &newTransform)
    {
        if (m_robotTransform)
          {
              // transform newTransform to QMatrix4x4
                QMatrix4x4 newMatrix;
                newMatrix.setToIdentity();
                newMatrix.translate(newTransform.translation().x()/1000.0, 0.0, -newTransform.translation().y()/1000.0);
                // Extract the rotation matrix
                Eigen::Matrix2d rotationMatrix = newTransform.rotation();
                // Compute the rotation angle in radians
                double angle = std::atan2(rotationMatrix(1, 0), rotationMatrix(0, 0));
                // Convert the angle to degrees
                double angleInDegrees = angle * (180.0 / M_PI);
                QMatrix3x3 rot;
                newMatrix.rotate(-90, 1, 0, 0);
                newMatrix.rotate(angleInDegrees, 0, 0, 1);
                m_robotTransform->setMatrix(newMatrix);
          }
    }

    void Viewer3D::createRoomTransform()
    {
        // create a room from inner_model data
        const double width = inner_model->room->get_width() /1000.0;
        const double depth = inner_model->room->get_depth() /1000.0;

        m_roomEntity = new Qt3DCore::QEntity(m_rootEntity);

        // can create walls as thin cuboids. You can extend this with proper colors and textures.
        auto *wallEntity = new Qt3DCore::QEntity(m_roomEntity);
        auto *wallMesh = new Qt3DExtras::QCuboidMesh();
        wallMesh->setXExtent(width);
        wallMesh->setYExtent(1.5); // height of the wall
        wallMesh->setZExtent(0.1);  // thin wall
        wallEntity->addComponent(wallMesh);
        auto *wallMat = new Qt3DExtras::QPhongMaterial(wallEntity);
        wallMat->setDiffuse(QColor(QRgb(0xaaaaaa)));
        wallEntity->addComponent(wallMat);
        auto *wallTransform = new Qt3DCore::QTransform();
        wallTransform->setTranslation(QVector3D(0, 0, -depth/2)); // adjust to form room boundaries
        wallEntity->addComponent(wallTransform);

        auto *wallEntity2 = new Qt3DCore::QEntity(m_roomEntity);
        auto *wallMesh2 = new Qt3DExtras::QCuboidMesh();
        wallMesh2->setXExtent(width);
        wallMesh2->setYExtent(1.5); // height of the wall
        wallMesh2->setZExtent(0.1);  // thin wall
        wallEntity2->addComponent(wallMesh2);
        auto *wallMat2 = new Qt3DExtras::QPhongMaterial(wallEntity2);
        wallMat2->setDiffuse(QColor(QRgb(0xaaaaaa)));
        wallEntity2->addComponent(wallMat2);
        auto *wallTransform2= new Qt3DCore::QTransform();
        wallTransform2->setRotationY(90);
        wallTransform2->setTranslation(QVector3D(width/2, 0, 0)); // adjust to form room boundaries
        wallEntity2->addComponent(wallTransform2);

        auto *wallEntity3 = new Qt3DCore::QEntity(m_roomEntity);
        auto *wallMesh3 = new Qt3DExtras::QCuboidMesh();
        wallMesh3->setXExtent(width);
        wallMesh3->setYExtent(1.5); // height of the wall
        wallMesh3->setZExtent(0.1);  // thin wall
        wallEntity3->addComponent(wallMesh3);
        auto *wallMat3 = new Qt3DExtras::QPhongMaterial(wallEntity3);
        wallMat3->setDiffuse(QColor(QRgb(0xaaaaaa)));
        wallEntity3->addComponent(wallMat3);
        auto *wallTransform3= new Qt3DCore::QTransform();
        wallTransform3->setTranslation(QVector3D(0, 0, depth/2)); // adjust to form room boundaries
        wallEntity3->addComponent(wallTransform3);

        auto *wallEntity4 = new Qt3DCore::QEntity(m_roomEntity);
        auto *wallMesh4 = new Qt3DExtras::QCuboidMesh();
        wallMesh4->setXExtent(width);
        wallMesh4->setYExtent(1.5); // height of the wall
        wallMesh4->setZExtent(0.1);  // thin wall
        wallEntity4->addComponent(wallMesh4);
        auto *wallMat4 = new Qt3DExtras::QPhongMaterial(wallEntity4);
        wallMat4->setDiffuse(QColor(QRgb(0xaaaaaa)));
        wallEntity4->addComponent(wallMat4);
        auto *wallTransform4= new Qt3DCore::QTransform();
        wallTransform4->setRotationY(-90);
        wallTransform4->setTranslation(QVector3D(-width/2, 0, 0)); // adjust to form room boundaries
        wallEntity4->addComponent(wallTransform4);
    }

    void Viewer3D::createFridgeTransform()
    {
        //qInfo() << __FILE__ << __FUNCTION__;
        auto fridge_params = inner_model->fridge->means;
        const double x = fridge_params(0);
        const double y = fridge_params(1);
        const double theta = fridge_params(2);
        const double width = fridge_params(3);
        const double depth = fridge_params(4);

        // --- Fridge: represented as a cube ---
        m_fridgeEntity = new Qt3DCore::QEntity(m_rootEntity);
        auto *fridgeMesh = new Qt3DExtras::QCuboidMesh();
        fridgeMesh->setXExtent(width);  // width
        fridgeMesh->setYExtent(1.7);  // height
        fridgeMesh->setZExtent(depth);  // depth
        m_fridgeEntity->addComponent(fridgeMesh);
        auto *fridgeMat = new Qt3DExtras::QPhongMaterial(m_fridgeEntity);
        fridgeMat->setDiffuse(QColor(QRgb(0xff0000)));
        m_fridgeEntity->addComponent(fridgeMat);
        m_fridgeTransform = new Qt3DCore::QTransform();
        m_fridgeTransform->setTranslation(QVector3D(x, 0.0, -y));  // Adjust to place the fridge in the room
        m_fridgeTransform->setRotationY(-theta);  // Rotate the fridge if needed.
        m_fridgeEntity->addComponent(m_fridgeTransform);
    }

    void Viewer3D::updateFridgeTransform()
    {
        auto fridge_params = inner_model->fridge->means;
        const double x = fridge_params(0);
        const double y = fridge_params(1);
        const double theta = fridge_params(2);
        const double width = fridge_params(3);
        const double depth = fridge_params(4);

        m_fridgeTransform->setTranslation(QVector3D(x, 0.0, -y));  // Adjust to place the fridge in the room
        m_fridgeTransform->setRotationY(-theta);  // Rotate the fridge if needed.
        auto r = m_fridgeEntity->componentsOfType<Qt3DExtras::QCuboidMesh>();  //TODO: save the pointer when creating the fridge
        r.first()->setXExtent(width);  // width
        r.first()->setZExtent(depth);  // depth
    }

    //////////////////////////////////////////////////////7
    void Viewer3D::add_lights()
    {
        auto *light = new Qt3DRender::QPointLight(m_rootEntity);
        light->setColor(Qt::white);
        light->setIntensity(5.0f);
        auto *lightTransform = new Qt3DCore::QTransform(m_rootEntity);
        lightTransform->setTranslation(QVector3D(0, 3, 0)); // Position the light above the scene.
        // Add the light and its transform as components to the root entity
        m_rootEntity->addComponent(light);  // Corrected line
        m_rootEntity->addComponent(lightTransform);  // Corrected line

        // --- Add a cone to visualize the point light ---
        // auto *lightConeEntity = new Qt3DCore::QEntity(m_rootEntity);
        // auto *lightConeMesh = new Qt3DExtras::QConeMesh();
        // lightConeMesh->setTopRadius(0.2f); // Adjust size as needed
        // lightConeMesh->setBottomRadius(0.0f);
        // lightConeMesh->setLength(0.5f); // Adjust size as needed
        // lightConeEntity->addComponent(lightConeMesh);
        // auto *lightConeMaterial = new Qt3DExtras::QPhongMaterial();
        // lightConeMaterial->setDiffuse(QColor(QRgb(0xffff00))); // Yellow color for the cone
        // lightConeEntity->addComponent(lightConeMaterial);
        // auto *lightConeTransform = new Qt3DCore::QTransform();
        // lightConeTransform->setTranslation(lightTransform->translation()); // Same position as the light
        // lightConeEntity->addComponent(lightConeTransform);

        //--- Add a Directional Light ---
        auto *directionalLight = new Qt3DRender::QDirectionalLight(m_rootEntity);
        directionalLight->setColor(Qt::white);
        directionalLight->setIntensity(5.5f); // Adjust intensity as needed.
        directionalLight->setWorldDirection(QVector3D(0, -1, 0)); // Direction from top down.
        auto *dlightTransform = new Qt3DCore::QTransform(m_rootEntity);
        dlightTransform->setTranslation(QVector3D(0, 3, 0)); // Position the light above the scene.
        // Add the directional light as a component to the root entity
        m_rootEntity->addComponent(directionalLight);
        m_rootEntity->addComponent(dlightTransform);
    }
    void Viewer3D::add_floor()
    {
        m_floorEntity = new Qt3DCore::QEntity(m_rootEntity);
        auto *floorMesh = new Qt3DExtras::QPlaneMesh();
        floorMesh->setWidth(10);
        floorMesh->setHeight(10);
        m_floorEntity->addComponent(floorMesh);
        // Load a grid texture (update the file path as needed or use a Qt resource)
        auto *textureLoader = new Qt3DRender::QTextureLoader(m_floorEntity);
        textureLoader->setSource(QUrl::fromLocalFile("meshes/grid.png")); // supply your grid texture file
        // Use a texture material to display the grid.
        auto *floorMaterial = new Qt3DExtras::QTextureMaterial(m_floorEntity);
        floorMaterial->setTexture(textureLoader);
        m_floorEntity->addComponent(floorMaterial);
        // Transform for the floor (if needed)
        auto *floorTransform = new Qt3DCore::QTransform(m_floorEntity);
        // Place floor at y=0.
        floorTransform->setTranslation(QVector3D(0.0f, 0.0f, 0.0f));
        m_floorEntity->addComponent(floorTransform);
    }
    void Viewer3D::add_robot()
    {
        m_robotEntity = new Qt3DCore::QEntity(m_rootEntity);
        auto *robotMesh = new Qt3DRender::QMesh();
        robotMesh->setSource(QUrl::fromLocalFile("meshes/shadow.obj"));
        m_robotEntity->addComponent(robotMesh);
        // Place robot at the center of the room
        m_robotTransform = new Qt3DCore::QTransform();
        m_robotTransform->setTranslation(QVector3D(0.0f, 0.0f, 0.0f));
        m_robotTransform->setRotationX(-90.0f);  // Rotate the robot if needed.
        m_robotTransform->setScale(1.f);
        m_robotEntity->addComponent(m_robotTransform);
        auto *robotMat = new Qt3DExtras::QPhongMaterial(m_robotEntity);
        robotMat->setDiffuse(QColor(QRgb(0x00ff00)));
        m_robotEntity->addComponent(robotMat);
    }
    void Viewer3D::add_axes()
    {
              // --- Frame Axes ---
        auto *frameAxesEntity = new Qt3DCore::QEntity(m_rootEntity);
        const float arrow_length = 1.f;
        // X-axis (red)
        auto *xAxisEntity = new Qt3DCore::QEntity(frameAxesEntity);
        auto *xAxisMesh = new Qt3DExtras::QCylinderMesh();
        xAxisMesh->setRadius(0.03f);
        xAxisMesh->setLength(arrow_length);
        xAxisEntity->addComponent(xAxisMesh);
        auto *xAxisMaterial = new Qt3DExtras::QPhongMaterial();
        xAxisMaterial->setDiffuse(QColor(QRgb(0xff0000))); // Red
        xAxisEntity->addComponent(xAxisMaterial);
        auto *xAxisTransform = new Qt3DCore::QTransform();
        xAxisTransform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(0, 0, 1), 90)); // Rotate to align with X-axis
        xAxisTransform->setTranslation(QVector3D(0.5, 0, 0));
        xAxisEntity->addComponent(xAxisTransform);
        // X-axis arrow
        Qt3DCore::QEntity *xAxisArrowEntity = new Qt3DCore::QEntity(xAxisEntity);
        Qt3DExtras::QConeMesh *xAxisArrowMesh = new Qt3DExtras::QConeMesh();
        xAxisArrowMesh->setTopRadius(0.1f);
        xAxisArrowMesh->setBottomRadius(0);
        xAxisArrowMesh->setLength(0.2f);
        xAxisArrowEntity->addComponent(xAxisArrowMesh);
        xAxisArrowEntity->addComponent(xAxisMaterial); // Reuse the same material
        Qt3DCore::QTransform *xAxisArrowTransform = new Qt3DCore::QTransform();
        xAxisArrowTransform->setTranslation(QVector3D(0.f, -0.5, 0)); // Position at the end of the axis
        xAxisArrowEntity->addComponent(xAxisArrowTransform);

        // Y-axis (green)
        auto *yAxisEntity = new Qt3DCore::QEntity(frameAxesEntity);
        auto *yAxisMesh = new Qt3DExtras::QCylinderMesh();
        yAxisMesh->setRadius(0.03f);
        yAxisMesh->setLength(arrow_length);
        yAxisEntity->addComponent(yAxisMesh);
        auto *yAxisMaterial = new Qt3DExtras::QPhongMaterial();
        yAxisMaterial->setDiffuse(QColor(QRgb(0x00ff00))); // Green
        yAxisEntity->addComponent(yAxisMaterial);
        auto *yAxisTransform = new Qt3DCore::QTransform();
        yAxisTransform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), 90)); // Rotate to align with Y-axis
        yAxisTransform->setTranslation(QVector3D(0, 0, -0.5));
        yAxisEntity->addComponent(yAxisTransform);
        // Y-axis arrow
        auto *yAxisArrowEntity = new Qt3DCore::QEntity(yAxisEntity);
        auto *yAxisArrowMesh = new Qt3DExtras::QConeMesh();
        yAxisArrowMesh->setTopRadius(0.1f);
        yAxisArrowMesh->setBottomRadius(0);
        yAxisArrowMesh->setLength(0.2f);
        yAxisArrowEntity->addComponent(yAxisArrowMesh);
        yAxisArrowEntity->addComponent(yAxisMaterial); // Reuse the same material
        auto *yAxisArrowTransform = new Qt3DCore::QTransform();
        yAxisArrowTransform->setTranslation(QVector3D(0, -0.5f, 0)); // Position at the end of the axis
        yAxisArrowEntity->addComponent(yAxisArrowTransform);
    }
    void Viewer3D::add_scene_camera()
    {
        // Load camera position
        const QSettings settings("YourOrganizationName", "YourAppName");
        auto position = settings.value("cameraPosition", QVector3D(0.0f, 5.0f, 5.0f)).value<QVector3D>(); // Default position if not found
        auto orientation = settings.value("cameraOrientation", QQuaternion()).value<QQuaternion>(); // Default orientation if not found

        // create cene Camera
        auto *camera = m_view->camera();
        camera->setProjectionType(Qt3DRender::QCameraLens::PerspectiveProjection);
        camera->lens()->setPerspectiveProjection(45.0f, 16.0f/9.0f, 0.1f, 10000.0f);
        camera->setViewCenter(QVector3D(0.0f, 0.0f, 0.0f));
        camera->setPosition(position);
        camera->transform()->setRotation(orientation); // Set loaded orientation

        // For camera controls
        auto *camController = new Qt3DExtras::QOrbitCameraController(m_rootEntity);
        camController->setCamera(camera);
    }
};  // namespace rc

