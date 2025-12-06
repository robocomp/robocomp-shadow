#include "qt3d_visualizer.h"
#include <Qt3DCore/QGeometry>
#include <Qt3DCore/QBuffer>
#include <Qt3DCore/QAttribute>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DRender/QGeometryRenderer>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QConeMesh>
#include <Qt3DExtras/QPlaneMesh>
#include <QFileInfo>
#include <cmath>

RoomVisualizer3D::RoomVisualizer3D(const QString &robotMeshPath)
    : view(nullptr)
      , rootEntity(nullptr)
      , roomEntity(nullptr)
      , robotEntity(nullptr)
      , uncertaintyEntity(nullptr)
      , robotTransform(nullptr)
      , roomTransform(nullptr)
      , uncertaintyTransform(nullptr)
      , robotMeshPath(robotMeshPath)
      , pointRadius(0.03f)
      , roomLineWidth(0.01f) // 10mm default line width
      , uncertaintyScale(1.0f) // 5x amplification for visibility
      , pointColor(Qt::red)
      , roomColor(Qt::blue)
      , robotColor(QColor(0, 200, 0))
      , uncertaintyColor(QColor(255, 165, 0, 100))
{
    // Create window
    view = new Qt3DExtras::Qt3DWindow();
    view->defaultFrameGraph()->setClearColor(QColor(50, 50, 60));
    view->setTitle("Room Detector 3D Visualization");
    view->resize(1280, 720);

    // Create root entity
    rootEntity = new Qt3DCore::QEntity();
    sceneEntity = new Qt3DCore::QEntity(rootEntity);

    setupScene();
    setupCamera();
    setupLighting();

    view->setRootEntity(rootEntity);
}

RoomVisualizer3D::~RoomVisualizer3D()
{
    if (view)
    {
        delete view;
    }
}

void RoomVisualizer3D::setupScene()
{
    // Create shared resources for point cloud (instancing)
    sharedPointMesh = new Qt3DExtras::QSphereMesh();
    sharedPointMesh->setRadius(pointRadius);
    sharedPointMesh->setRings(8);
    sharedPointMesh->setSlices(8);

    sharedPointMaterial = new Qt3DExtras::QPhongMaterial();
    sharedPointMaterial->setDiffuse(pointColor);
    sharedPointMaterial->setAmbient(pointColor.darker(150));
    sharedPointMaterial->setSpecular(Qt::white);
    sharedPointMaterial->setShininess(50.0f);

    // Create coordinate axes
    createCoordinateAxes();

    // Create ground grid
    createGroundGrid();
}

void RoomVisualizer3D::setupCamera()
{
    camera = view->camera();
    camera->lens()->setPerspectiveProjection(45.0f, 16.0f / 9.0f, 0.1f, 1000.0f);
    camera->setPosition(QVector3D(8, 8, 6));
    camera->setViewCenter(QVector3D(0, 0, 0));
    camera->setUpVector(QVector3D(0, 0, 1));

    // Camera controller
    camController = new CustomCameraController(rootEntity);
    camController->setCamera(camera);
    camController->setRotationSpeed(0.005f);
    camController->setPanSpeed(0.01f);
    camController->setZoomSpeed(0.15f);
    camController->setTarget(QVector3D(0, 0, 0));
}

void RoomVisualizer3D::setupLighting()
{
    // Main light
    auto *const lightEntity = new Qt3DCore::QEntity(rootEntity);
    auto *const light = new Qt3DRender::QPointLight(lightEntity);
    light->setColor(Qt::white);
    light->setIntensity(1.2f);
    lightEntity->addComponent(light);

    auto *const lightTransform = new Qt3DCore::QTransform(lightEntity);
    lightTransform->setTranslation(QVector3D(5, 5, 10));
    lightEntity->addComponent(lightTransform);

    // Fill light
    auto *const fillLightEntity = new Qt3DCore::QEntity(rootEntity);
    auto *const fillLight = new Qt3DRender::QPointLight(fillLightEntity);
    fillLight->setColor(Qt::white);
    fillLight->setIntensity(0.6f);
    fillLightEntity->addComponent(fillLight);

    auto *const fillLightTransform = new Qt3DCore::QTransform(fillLightEntity);
    fillLightTransform->setTranslation(QVector3D(-5, -5, 5));
    fillLightEntity->addComponent(fillLightTransform);

    // Ambient light for better visibility
    auto *const ambientLightEntity = new Qt3DCore::QEntity(rootEntity);
    auto *const ambientLight = new Qt3DRender::QPointLight(ambientLightEntity);
    ambientLight->setColor(QColor(200, 200, 220));
    ambientLight->setIntensity(0.3f);
    ambientLightEntity->addComponent(ambientLight);

    auto *const ambientTransform = new Qt3DCore::QTransform(ambientLightEntity);
    ambientTransform->setTranslation(QVector3D(0, 0, 15));
    ambientLightEntity->addComponent(ambientTransform);
}

void RoomVisualizer3D::createCoordinateAxes()
{
    axesEntity = new Qt3DCore::QEntity(sceneEntity);

    auto createAxis = [this](const QVector3D &direction, const QColor &color, float length = 0.2f)
    {
        // Cylinder for shaft
        const auto cylinder = new Qt3DExtras::QCylinderMesh();
        cylinder->setRadius(0.005f);
        cylinder->setLength(length);

        const auto material = new Qt3DExtras::QPhongMaterial();
        material->setDiffuse(color);
        material->setAmbient(color.darker(120));
        material->setSpecular(Qt::white);
        material->setShininess(80.0f);

        const auto entity = new Qt3DCore::QEntity(axesEntity);
        entity->addComponent(cylinder);
        entity->addComponent(material);

        const auto transform = new Qt3DCore::QTransform();
        const QVector3D pos = direction * (length / 2.0f);
        transform->setTranslation(pos);

        // Rotate cylinder to align with direction (default cylinder is Y-up in Qt3D)
        if (direction.x() != 0)
        {
            // X axis: rotate 90° around Z
            transform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(0, 0, 1), 90));
        } else if (direction.z() != 0)
        {
            // Z axis: rotate 90° around X
            transform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), 90));
        }
        // Y axis needs no rotation (cylinder default is Y-up)

        entity->addComponent(transform);

        // Arrow head
        const auto cone = new Qt3DExtras::QConeMesh();
        cone->setBottomRadius(0.03f);
        cone->setLength(0.1f);

        const auto coneEntity = new Qt3DCore::QEntity(axesEntity);
        coneEntity->addComponent(cone);
        coneEntity->addComponent(material);

        auto *const coneTransform = new Qt3DCore::QTransform();
        const QVector3D conePos = direction * (length + 0.125f);

        // Same rotation logic for cone
        if (direction.x() != 0)
        {
            coneTransform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(0, 0, 1), 90));
        } else if (direction.z() != 0)
        {
            coneTransform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), 90));
        }

        coneTransform->setTranslation(conePos);
        coneEntity->addComponent(coneTransform);
    };

    createAxis(QVector3D(1, 0, 0), Qt::green); // X axis - RED (right in robotics)
    createAxis(QVector3D(0, 1, 0), Qt::red); // Y axis - GREEN (forward in robotics)
    createAxis(QVector3D(0, 0, 1), Qt::blue); // Z axis - BLUE (up)
}

void RoomVisualizer3D::createGroundGrid()
{
    groundEntity = new Qt3DCore::QEntity(sceneEntity);

    // Grid parameters
    const float gridSize = 20.0f;
    const float gridStep = 1.0f;
    const int numLines = static_cast<int>(gridSize / gridStep) + 1;

    // Create geometry for grid lines
    auto *const geometry = new Qt3DCore::QGeometry(groundEntity);

    // Calculate number of vertices (2 per line, numLines in each direction)
    const int vertexCount = numLines * 4; // 2 vertices per line, numLines in X and Y
    QByteArray bufferBytes;
    bufferBytes.resize(vertexCount * 3 * sizeof(float)); // 3 floats per vertex (x, y, z)
    float *const positions = reinterpret_cast<float *>(bufferBytes.data());

    int idx = 0;
    const float halfSize = gridSize / 2.0f;

    // Lines parallel to X axis
    for (int i = 0; i < numLines; ++i)
    {
        const float y = -halfSize + i * gridStep;
        // Start point
        positions[idx++] = -halfSize;
        positions[idx++] = y;
        positions[idx++] = 0.0f;
        // End point
        positions[idx++] = halfSize;
        positions[idx++] = y;
        positions[idx++] = 0.0f;
    }

    // Lines parallel to Y axis
    for (int i = 0; i < numLines; ++i)
    {
        const float x = -halfSize + i * gridStep;
        // Start point
        positions[idx++] = x;
        positions[idx++] = -halfSize;
        positions[idx++] = 0.0f;
        // End point
        positions[idx++] = x;
        positions[idx++] = halfSize;
        positions[idx++] = 0.0f;
    }

    // Create buffer
    auto *const buf = new Qt3DCore::QBuffer(geometry);
    buf->setData(bufferBytes);

    // Position attribute
    auto *const positionAttribute = new Qt3DCore::QAttribute(geometry);
    positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
    positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
    positionAttribute->setVertexSize(3);
    positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
    positionAttribute->setBuffer(buf);
    positionAttribute->setByteStride(3 * sizeof(float));
    positionAttribute->setCount(vertexCount);

    geometry->addAttribute(positionAttribute);

    // Create geometry renderer
    auto *const lineRenderer = new Qt3DRender::QGeometryRenderer(groundEntity);
    lineRenderer->setGeometry(geometry);
    lineRenderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);

    // Material for grid lines
    auto *const material = new Qt3DExtras::QPhongMaterial();
    material->setDiffuse(QColor(100, 100, 120));
    material->setAmbient(QColor(80, 80, 100));
    material->setSpecular(QColor(0, 0, 0));

    groundEntity->addComponent(lineRenderer);
    groundEntity->addComponent(material);
}

void RoomVisualizer3D::updatePointCloud(const std::vector<Eigen::Vector3f> &points)
{
    if (points.empty())
    {
        // Hide existing point cloud if any
        if (!pointEntities.empty() && pointEntities[0])
            pointEntities[0]->setEnabled(false);
        return;
    }

    // Prepare position data
    QByteArray positionData;
    positionData.resize(static_cast<int>(points.size() * 3 * sizeof(float)));
    float *posPtr = reinterpret_cast<float *>(positionData.data());

    for (const auto &p : points)
    {
        *posPtr++ = p.x();
        *posPtr++ = p.y();
        *posPtr++ = p.z();
    }

    // Reuse existing entity or create new one
    if (pointEntities.empty() || !pointEntities[0])
    {
        // First time: create entity and all components
        auto *const entity = new Qt3DCore::QEntity(sceneEntity);

        auto *geometry = new Qt3DCore::QGeometry(entity);

        auto *positionBuffer = new Qt3DCore::QBuffer(geometry);
        positionBuffer->setObjectName("pointCloudBuffer");
        positionBuffer->setData(positionData);

        auto *positionAttribute = new Qt3DCore::QAttribute(geometry);
        positionAttribute->setName(Qt3DCore::QAttribute::defaultPositionAttributeName());
        positionAttribute->setVertexBaseType(Qt3DCore::QAttribute::Float);
        positionAttribute->setVertexSize(3);
        positionAttribute->setAttributeType(Qt3DCore::QAttribute::VertexAttribute);
        positionAttribute->setBuffer(positionBuffer);
        positionAttribute->setByteStride(3 * sizeof(float));
        positionAttribute->setCount(static_cast<uint>(points.size()));

        geometry->addAttribute(positionAttribute);

        auto *geometryRenderer = new Qt3DRender::QGeometryRenderer(entity);
        geometryRenderer->setObjectName("pointCloudRenderer");
        geometryRenderer->setGeometry(geometry);
        geometryRenderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Points);
        geometryRenderer->setVertexCount(static_cast<int>(points.size()));

        auto *material = new Qt3DExtras::QPhongMaterial(entity);
        material->setDiffuse(pointColor);
        material->setAmbient(pointColor);

        entity->addComponent(geometryRenderer);
        entity->addComponent(material);

        pointEntities.clear();
        pointEntities.push_back(entity);
    }
    else
    {
        // Reuse existing: just update buffer and count
        auto *entity = pointEntities[0];
        entity->setEnabled(true);

        // Find the geometry renderer and update it
        for (auto *component : entity->components())
        {
            if (auto *renderer = qobject_cast<Qt3DRender::QGeometryRenderer *>(component))
            {
                renderer->setVertexCount(static_cast<int>(points.size()));

                // Update buffer data
                auto *geometry = renderer->geometry();
                for (auto *attr : geometry->attributes())
                {
                    if (attr->name() == Qt3DCore::QAttribute::defaultPositionAttributeName())
                    {
                        attr->buffer()->setData(positionData);
                        attr->setCount(static_cast<uint>(points.size()));
                        break;
                    }
                }
                break;
            }
        }
    }
}

void RoomVisualizer3D::updatePointCloud(const RoboCompLidar3D::TPoints &points)
{
    // std::vector<Eigen::Vector2f> epoints;
    // epoints.reserve(points.size());
    // for (const auto &p: points)
    //     epoints.emplace_back(p.x / 1000.0f, p.y / 1000.0f); // Convert mm to meters
    // updatePointCloud(epoints);
}

Qt3DCore::QEntity *RoomVisualizer3D::createLineBox(float width, float height, const QColor &color)
{
    auto *const entity = new Qt3DCore::QEntity(sceneEntity);

    // Create wireframe box using thin cylinders for edges
    auto createEdge = [&](const QVector3D &start, const QVector3D &end)
    {
        auto *const cylinder = new Qt3DExtras::QCylinderMesh();
        const float length = (end - start).length();
        cylinder->setRadius(roomLineWidth); // Use configurable line width
        cylinder->setLength(length);

        auto *const material = new Qt3DExtras::QPhongMaterial();
        material->setDiffuse(color);
        material->setAmbient(color.darker(120));
        material->setSpecular(Qt::white);
        material->setShininess(60.0f);

        auto *const edgeEntity = new Qt3DCore::QEntity(entity);
        edgeEntity->addComponent(cylinder);
        edgeEntity->addComponent(material);

        auto *const transform = new Qt3DCore::QTransform();
        const QVector3D center = (start + end) / 2.0f;
        transform->setTranslation(center);

        // Rotate to align with edge direction
        const QVector3D direction = (end - start).normalized();
        const QVector3D defaultDir(0, 1, 0); // Cylinder default is Y-up

        const QVector3D axis = QVector3D::crossProduct(defaultDir, direction);
        const float angle = qRadiansToDegrees(qAcos(QVector3D::dotProduct(defaultDir, direction)));

        if (axis.length() > 0.001f)
        {
            transform->setRotation(QQuaternion::fromAxisAndAngle(axis.normalized(), angle));
        } else if (angle > 90.0f)
        {
            // 180 degree flip
            transform->setRotation(QQuaternion::fromAxisAndAngle(QVector3D(1, 0, 0), 180));
        }

        edgeEntity->addComponent(transform);
    };

    const float hw = width / 2.0f;
    const float hh = height / 2.0f;
    const float z = 1.0f; // Room height for visualization

    // Bottom rectangle (on ground)
    createEdge(QVector3D(-hw, -hh, 0), QVector3D(hw, -hh, 0));
    createEdge(QVector3D(hw, -hh, 0), QVector3D(hw, hh, 0));
    createEdge(QVector3D(hw, hh, 0), QVector3D(-hw, hh, 0));
    createEdge(QVector3D(-hw, hh, 0), QVector3D(-hw, -hh, 0));

    // Top rectangle
    createEdge(QVector3D(-hw, -hh, z), QVector3D(hw, -hh, z));
    createEdge(QVector3D(hw, -hh, z), QVector3D(hw, hh, z));
    createEdge(QVector3D(hw, hh, z), QVector3D(-hw, hh, z));
    createEdge(QVector3D(-hw, hh, z), QVector3D(-hw, -hh, z));

    // Vertical edges
    createEdge(QVector3D(-hw, -hh, 0), QVector3D(-hw, -hh, z));
    createEdge(QVector3D(hw, -hh, 0), QVector3D(hw, -hh, z));
    createEdge(QVector3D(hw, hh, 0), QVector3D(hw, hh, z));
    createEdge(QVector3D(-hw, hh, 0), QVector3D(-hw, hh, z));

    return entity;
}

void RoomVisualizer3D::updateRoom(float half_width, float half_depth)
{
    // Create room entity only once
    if (!roomEntity)
    {
        roomEntity = createLineBox(1.0f, 1.0f, roomColor); // Create unit box (1x1)

        // Add transform component
        roomTransform = new Qt3DCore::QTransform();
        roomEntity->addComponent(roomTransform);
    }

    // Just update the scale to match room dimensions (no recreation!)
    roomTransform->setScale3D(QVector3D(half_width * 2.0f, half_depth * 2.0f, 2.5f));
}

void RoomVisualizer3D::updateRobotPose(float x, float y, float theta)
{
    // Create robot entity if it doesn't exist
    if (!robotEntity)
    {
        robotEntity = new Qt3DCore::QEntity(sceneEntity);

        auto *const material = new Qt3DExtras::QPhongMaterial();
        material->setDiffuse(robotColor);
        material->setAmbient(robotColor.darker(130));
        material->setSpecular(Qt::white);
        material->setShininess(100.0f);

        robotTransform = new Qt3DCore::QTransform();

        // Try to load .obj mesh first
        const QFileInfo meshFile(robotMeshPath);
        qDebug() << "Attempting to load robot mesh:";
        qDebug() << "  Path:" << robotMeshPath;
        qDebug() << "  Absolute path:" << meshFile.absoluteFilePath();
        qDebug() << "  Exists:" << meshFile.exists();

        if (meshFile.exists())
        {
            robotMesh = new Qt3DRender::QMesh();
            const QUrl meshUrl = QUrl::fromLocalFile(meshFile.absoluteFilePath());
            qDebug() << "  Loading from URL:" << meshUrl;
            robotMesh->setSource(meshUrl);
            robotEntity->addComponent(robotMesh);

            // Check mesh status
            QObject::connect(robotMesh, &Qt3DRender::QMesh::statusChanged,
                             [](Qt3DRender::QMesh::Status status)
                             {
                                 qDebug() << "Mesh status changed to:" << status;
                                 if (status == Qt3DRender::QMesh::Error)
                                     qDebug() << "ERROR: Failed to load mesh!";
                             });
        } else
        {
            qDebug() << "  Mesh file not found! Using fallback box.";
            // Use fallback box
            fallbackRobotMesh = new Qt3DExtras::QCuboidMesh();
            fallbackRobotMesh->setXExtent(0.46f); // 460mm
            fallbackRobotMesh->setYExtent(0.48f); // 480mm
            fallbackRobotMesh->setZExtent(0.3f); // 300mm height
            robotEntity->addComponent(fallbackRobotMesh);
        }
        robotEntity->addComponent(material);
        robotEntity->addComponent(robotTransform);
    }

    // Update robot pose
    robotTransform->setTranslation(QVector3D(x, y, 0.15f)); // Raise slightly above ground
    robotTransform->setRotationZ(qRadiansToDegrees(theta));

    // Scale: try different scales depending on what's loaded
    if (robotMesh)
        robotTransform->setScale(1.0f); // Try 1:1 first, adjust if needed
    else
        robotTransform->setScale(1.0f); // Box is already in meters
}

// Replace the draw_door() method in qt3d_visualizer.cpp with this implementation
// Coordinate system: X=right, Y=forward, Z=up (aligned with DoorModel)

void RoomVisualizer3D::draw_door(float x, float y, float z, float theta, float width, float height, float open_angle)
{
    // // Transform: Robot(x,y) -> Vis(y, -x), theta -> theta - 90°  TODO: CHECK THIS
    //const float vis_x = -y;
    //const float vis_y = x;
    const float vis_x = x;
    const float vis_y = y;
    const float vis_theta = theta;// + static_cast<float>(M_PI_2);

    const float theta_deg = vis_theta * 180.0f / static_cast<float>(M_PI);
    const float open_deg = open_angle * 180.0f / static_cast<float>(M_PI);

    // Frame parameters (match DoorModel)
    const float frame_thickness = 0.10f;
    const float frame_depth = 0.15f;
    const float leaf_thickness = 0.04f;

    // If door already exists, just update transforms
    if (doorEntity and doorTransform_ and hingeTransform_)
    {
        // Update global door pose
        doorTransform_->setTranslation(QVector3D(vis_x, vis_y, z));
        doorTransform_->setRotationZ(theta_deg);

        // Update leaf opening angle
        hingeTransform_->setRotationZ(open_deg);

        // Note: if width/height change significantly, we'd need to recreate
        // For now, assume geometry is stable after initialization
        return;
    }

    // First time: create the full hierarchy
    doorEntity = new Qt3DCore::QEntity(sceneEntity);

    // Global transform
    doorTransform_ = new Qt3DCore::QTransform();
    doorTransform_->setTranslation(QVector3D(vis_x, vis_y, z));
    doorTransform_->setRotationZ(theta_deg);
    doorEntity->addComponent(doorTransform_);

    // Materials
    auto *material_frame = new Qt3DExtras::QPhongMaterial();
    material_frame->setDiffuse(QColor(139, 90, 43));
    material_frame->setAmbient(QColor(80, 50, 25));

    auto *material_leaf = new Qt3DExtras::QPhongMaterial();
    material_leaf->setDiffuse(QColor(180, 140, 100));
    material_leaf->setAmbient(QColor(100, 80, 60));

    // ==================== FRAME ====================

    // Left jamb
    auto *leftJambMesh = new Qt3DExtras::QCuboidMesh();
    leftJambMesh->setXExtent(frame_thickness);
    leftJambMesh->setYExtent(frame_depth);
    leftJambMesh->setZExtent(height);

    auto *leftJambEntity = new Qt3DCore::QEntity(doorEntity);
    auto *leftJambTransform = new Qt3DCore::QTransform();
    leftJambTransform->setTranslation(QVector3D(-width/2.0f - frame_thickness/2.0f, 0.0f, height/2.0f));
    leftJambEntity->addComponent(leftJambMesh);
    leftJambEntity->addComponent(leftJambTransform);
    leftJambEntity->addComponent(material_frame);

    // Right jamb
    auto *rightJambMesh = new Qt3DExtras::QCuboidMesh();
    rightJambMesh->setXExtent(frame_thickness);
    rightJambMesh->setYExtent(frame_depth);
    rightJambMesh->setZExtent(height);

    auto *rightJambEntity = new Qt3DCore::QEntity(doorEntity);
    auto *rightJambTransform = new Qt3DCore::QTransform();
    rightJambTransform->setTranslation(QVector3D(width/2.0f + frame_thickness/2.0f, 0.0f, height/2.0f));
    rightJambEntity->addComponent(rightJambMesh);
    rightJambEntity->addComponent(rightJambTransform);
    rightJambEntity->addComponent(material_frame);

    // Top lintel
    auto *lintelMesh = new Qt3DExtras::QCuboidMesh();
    lintelMesh->setXExtent(width + 2.0f * frame_thickness);
    lintelMesh->setYExtent(frame_depth);
    lintelMesh->setZExtent(frame_thickness);

    auto *lintelEntity = new Qt3DCore::QEntity(doorEntity);
    auto *lintelTransform = new Qt3DCore::QTransform();
    lintelTransform->setTranslation(QVector3D(0.0f, 0.0f, height + frame_thickness/2.0f));
    lintelEntity->addComponent(lintelMesh);
    lintelEntity->addComponent(lintelTransform);
    lintelEntity->addComponent(material_frame);

    // ==================== LEAF ====================

    auto *hingeEntity = new Qt3DCore::QEntity(doorEntity);
    hingeTransform_ = new Qt3DCore::QTransform();
    hingeTransform_->setTranslation(QVector3D(-width/2.0f, 0.0f, 0.0f));
    hingeTransform_->setRotationZ(open_deg);
    hingeEntity->addComponent(hingeTransform_);

    auto *leafMesh = new Qt3DExtras::QCuboidMesh();
    leafMesh->setXExtent(width);
    leafMesh->setYExtent(leaf_thickness);
    leafMesh->setZExtent(height);

    auto *leafEntity = new Qt3DCore::QEntity(hingeEntity);
    auto *leafTransform = new Qt3DCore::QTransform();
    leafTransform->setTranslation(QVector3D(width/2.0f, 0.0f, height/2.0f));
    leafEntity->addComponent(leafMesh);
    leafEntity->addComponent(leafTransform);
    leafEntity->addComponent(material_leaf);
}

//////////////////////////////////////////////////////////////////////////////////
Qt3DCore::QEntity *RoomVisualizer3D::createUncertaintyEllipse(float std_x, float std_y, const QColor &color)
{
    const auto entity = new Qt3DCore::QEntity(sceneEntity);

    // Create ellipse mesh (unit sphere that will be scaled)
    const auto ellipseMesh = new Qt3DExtras::QSphereMesh();
    ellipseMesh->setRadius(1.0f);
    ellipseMesh->setRings(16);
    ellipseMesh->setSlices(16);

    const auto material = new Qt3DExtras::QPhongMaterial();

    // Set color with alpha channel for transparency
    QColor transparentColor = color;
    transparentColor.setAlphaF(0.3f);

    material->setDiffuse(transparentColor);
    material->setAmbient(transparentColor.darker(120));
    material->setSpecular(Qt::white);
    material->setShininess(20.0f);

    // Create transform (will be updated externally)
    const auto transform = new Qt3DCore::QTransform();
    transform->setScale3D(QVector3D(std_x, std_y, 0.01f));
    transform->setTranslation(QVector3D(0, 0, 0.05f));

    entity->addComponent(ellipseMesh);
    entity->addComponent(material);
    entity->addComponent(transform);

    return entity;
}

void RoomVisualizer3D::updateUncertainty(float pos_std_x, float pos_std_y, float theta_std)
{
    // Create uncertainty entity only once
    if (not uncertaintyEntity)
    {
        uncertaintyEntity = createUncertaintyEllipse(1.0f, 1.0f, uncertaintyColor); // Unit ellipse
        // Get the transform component that was added in createUncertaintyEllipse
        for (auto *const component: uncertaintyEntity->components())
            if (auto *transform = qobject_cast<Qt3DCore::QTransform *>(component))
            {
                uncertaintyTransform = transform;
                break;
            }
    }

    // Apply scale factor for visibility (2-sigma = 95% confidence)
    float scale_x = pos_std_x * 2.0f * uncertaintyScale;
    float scale_y = pos_std_y * 2.0f * uncertaintyScale;

    // Minimum size for visibility
    scale_x = std::max(scale_x, 0.05f);
    scale_y = std::max(scale_y, 0.05f);

    if (uncertaintyTransform)
    {
        uncertaintyTransform->setScale3D(QVector3D(scale_x, scale_y, 0.01f));
        // Position at robot's current location (get from robot transform)
        if (robotTransform)
        {
            const QVector3D robotPos = robotTransform->translation();
            uncertaintyTransform->setTranslation(QVector3D(robotPos.x(), robotPos.y(), 0.02f));
        } else
            uncertaintyTransform->setTranslation(QVector3D(0, 0, 0.02f));
    }

    // Ensure it's visible
    if (uncertaintyEntity)
        uncertaintyEntity->setEnabled(true);
}

void RoomVisualizer3D::showPointCloud(bool visible)
{
    for (auto *const entity: pointEntities)
        entity->setEnabled(visible);
}

void RoomVisualizer3D::showRoom(bool visible)
{
    if (roomEntity)
        roomEntity->setEnabled(visible);
}

void RoomVisualizer3D::showRobot(bool visible)
{
    if (robotEntity)
        robotEntity->setEnabled(visible);
}

void RoomVisualizer3D::showUncertainty(bool visible)
{
    if (uncertaintyEntity)
        uncertaintyEntity->setEnabled(visible);
}

void RoomVisualizer3D::resetCamera()
{
    camera->setPosition(QVector3D(8, 8, 6));
    camera->setViewCenter(QVector3D(0, 0, 0));
}

void RoomVisualizer3D::setCameraDistance(float distance)
{
    const QVector3D direction = (camera->position() - camera->viewCenter()).normalized();
    camera->setPosition(camera->viewCenter() + direction * distance);
}

void RoomVisualizer3D::show()
{
    view->show();
}

void RoomVisualizer3D::hide()
{
    view->hide();
}

void RoomVisualizer3D::setRobotVisibility(bool visible)
{
    if (robotEntity)
        robotEntity->setEnabled(visible);
}

void RoomVisualizer3D::forceRobotBox()
{
    // Remove existing robot if any
    if (robotEntity)
    {
        robotEntity->setParent((Qt3DCore::QNode *) nullptr);
        robotEntity->deleteLater();
        robotEntity = nullptr;
        robotMesh = nullptr;
        fallbackRobotMesh = nullptr;
        robotTransform = nullptr;
    }

    // Create robot with box only
    robotEntity = new Qt3DCore::QEntity(sceneEntity);

    fallbackRobotMesh = new Qt3DExtras::QCuboidMesh();
    fallbackRobotMesh->setXExtent(0.46f);
    fallbackRobotMesh->setYExtent(0.48f);
    fallbackRobotMesh->setZExtent(0.3f);

    auto *const material = new Qt3DExtras::QPhongMaterial();
    material->setDiffuse(robotColor);
    material->setAmbient(robotColor.darker(130));
    material->setSpecular(Qt::white);
    material->setShininess(100.0f);

    robotTransform = new Qt3DCore::QTransform();
    robotTransform->setTranslation(QVector3D(0, 0, 0.15f));

    robotEntity->addComponent(fallbackRobotMesh);
    robotEntity->addComponent(material);
    robotEntity->addComponent(robotTransform);

    qDebug() << "Forced robot box created at origin for testing";
}

void RoomVisualizer3D::setRoomLineWidth(float width)
{
    roomLineWidth = width;

    // If room already exists, need to recreate it with new line width
    if (roomEntity)
    {
        // Remove old room
        roomEntity->setParent((Qt3DCore::QNode *) nullptr);
        roomEntity->deleteLater();
        roomEntity = nullptr;
        roomTransform = nullptr;

        // Recreate with new line width (will happen on next updateRoom call)
        qDebug() << "Room line width changed to" << width << "meters. Room will be recreated on next update.";
    }
}

void RoomVisualizer3D::setUncertaintyScale(float scale)
{
    uncertaintyScale = scale;
    qDebug() << "Uncertainty scale factor set to" << scale << "x";
}