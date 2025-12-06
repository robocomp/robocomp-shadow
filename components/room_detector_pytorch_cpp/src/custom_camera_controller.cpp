/*
 * Custom Camera Controller for Qt3D
 */

#include "custom_camera_controller.h"
#include <QMouseEvent>
#include <QWheelEvent>
#include <QCoreApplication>
#include <cmath>

CustomCameraController::CustomCameraController(Qt3DCore::QNode *parent)
    : Qt3DCore::QEntity(parent)
{
    // Install event filter on application to catch mouse events
    QCoreApplication::instance()->installEventFilter(this);
}

CustomCameraController::~CustomCameraController()
{
    QCoreApplication::instance()->removeEventFilter(this);
}

void CustomCameraController::setCamera(Qt3DRender::QCamera *camera)
{
    if (m_camera != camera)
    {
        m_camera = camera;
        if (m_camera)
        {
            // Initialize camera state from current camera position
            QVector3D pos = m_camera->position();
            QVector3D dir = m_target - pos;
            m_distance = dir.length();
            if (m_distance > 0.01f)
            {
                dir.normalize();
                m_azimuth = std::atan2(dir.x(), dir.y());
                m_elevation = std::asin(-dir.z());
            }
            updateCameraPosition();
        }
        emit cameraChanged();
    }
}

void CustomCameraController::setRotationSpeed(float speed)
{
    if (!qFuzzyCompare(m_rotationSpeed, speed))
    {
        m_rotationSpeed = speed;
        emit rotationSpeedChanged();
    }
}

void CustomCameraController::setPanSpeed(float speed)
{
    if (!qFuzzyCompare(m_panSpeed, speed))
    {
        m_panSpeed = speed;
        emit panSpeedChanged();
    }
}

void CustomCameraController::setZoomSpeed(float speed)
{
    if (!qFuzzyCompare(m_zoomSpeed, speed))
    {
        m_zoomSpeed = speed;
        emit zoomSpeedChanged();
    }
}

void CustomCameraController::setTarget(const QVector3D &target)
{
    m_target = target;
    updateCameraPosition();
}

void CustomCameraController::resetView()
{
    m_target = QVector3D(0.0f, 0.0f, 0.0f);
    m_distance = 10.0f;
    m_azimuth = 0.0f;
    m_elevation = 0.5f;
    updateCameraPosition();
}

void CustomCameraController::updateCameraPosition()
{
    if (!m_camera)
        return;

    // Clamp values
    m_distance = qBound(m_minDistance, m_distance, m_maxDistance);
    m_elevation = qBound(m_minElevation, m_elevation, m_maxElevation);

    // Convert spherical to Cartesian coordinates
    // elevation: 0 = horizontal, positive = looking down from above
    float cosElev = std::cos(m_elevation);
    float sinElev = std::sin(m_elevation);
    float cosAzim = std::cos(m_azimuth);
    float sinAzim = std::sin(m_azimuth);

    // Camera position relative to target
    // Y+ is forward, X+ is right, Z+ is up
    QVector3D offset(
        m_distance * cosElev * sinAzim,   // X
        -m_distance * cosElev * cosAzim,  // Y (negative to look at target from behind)
        m_distance * sinElev              // Z
    );

    QVector3D cameraPos = m_target + offset;

    m_camera->setPosition(cameraPos);
    m_camera->setViewCenter(m_target);
    m_camera->setUpVector(QVector3D(0.0f, 0.0f, 1.0f));
}

bool CustomCameraController::eventFilter(QObject *obj, QEvent *event)
{
    if (!m_camera)
        return QObject::eventFilter(obj, event);

    switch (event->type())
    {
    case QEvent::MouseButtonPress:
    {
        auto *mouseEvent = static_cast<QMouseEvent *>(event);
        m_lastMousePos = mouseEvent->pos();

        if (mouseEvent->button() == Qt::LeftButton)
            m_leftButtonPressed = true;
        else if (mouseEvent->button() == Qt::MiddleButton)
            m_middleButtonPressed = true;
        else if (mouseEvent->button() == Qt::RightButton)
            m_rightButtonPressed = true;

        return false;  // Don't consume, let Qt3D also handle
    }

    case QEvent::MouseButtonRelease:
    {
        auto *mouseEvent = static_cast<QMouseEvent *>(event);

        if (mouseEvent->button() == Qt::LeftButton)
            m_leftButtonPressed = false;
        else if (mouseEvent->button() == Qt::MiddleButton)
            m_middleButtonPressed = false;
        else if (mouseEvent->button() == Qt::RightButton)
            m_rightButtonPressed = false;

        return false;
    }

    case QEvent::MouseMove:
    {
        auto *mouseEvent = static_cast<QMouseEvent *>(event);
        QPoint delta = mouseEvent->pos() - m_lastMousePos;
        m_lastMousePos = mouseEvent->pos();

        if (m_leftButtonPressed)
        {
            // Rotate/orbit around target
            m_azimuth -= delta.x() * m_rotationSpeed;
            m_elevation += delta.y() * m_rotationSpeed;
            updateCameraPosition();
            return true;  // Consume event
        }
        else if (m_rightButtonPressed)
        {
            // Pan - move target in camera's local XY plane
            // Moving mouse right should drag scene right (move target left)
            // Moving mouse up should drag scene up (move target down)
            QVector3D right = QVector3D::crossProduct(
                m_camera->upVector(),
                (m_target - m_camera->position()).normalized()
            ).normalized();
            QVector3D up = m_camera->upVector();

            float panScale = m_panSpeed * m_distance;  // Scale pan by distance
            m_target += right * (delta.x() * panScale);
            m_target += up * (-delta.y() * panScale);
            updateCameraPosition();
            return true;
        }

        return false;
    }

    case QEvent::Wheel:
    {
        auto *wheelEvent = static_cast<QWheelEvent *>(event);
        float delta = wheelEvent->angleDelta().y();

        // Zoom in/out
        float zoomFactor = 1.0f - (delta / 1200.0f) * m_zoomSpeed;
        m_distance *= zoomFactor;
        updateCameraPosition();

        return true;  // Consume wheel events
    }

    default:
        break;
    }

    return QObject::eventFilter(obj, event);
}