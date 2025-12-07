/*
 * Custom Camera Controller for Qt3D
 * Mimics Open3D/Webots camera behavior:
 * - Left mouse: Rotate/orbit around target
 * - Middle mouse: Pan
 * - Right mouse: Zoom (drag) or use scroll wheel
 * - Scroll wheel: Zoom
 */

#ifndef CUSTOM_CAMERA_CONTROLLER_H
#define CUSTOM_CAMERA_CONTROLLER_H

#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QCamera>
#include <QVector3D>
#include <QPoint>
#include <QtMath>
#include <Qt3DInput/QMouseDevice>
#include <Qt3DInput/QMouseHandler>
#include <QEvent>
#include <QMouseEvent>

class CustomCameraController : public Qt3DCore::QEntity
{
    Q_OBJECT
    Q_PROPERTY(Qt3DRender::QCamera *camera READ camera WRITE setCamera NOTIFY cameraChanged)
    Q_PROPERTY(float rotationSpeed READ rotationSpeed WRITE setRotationSpeed NOTIFY rotationSpeedChanged)
    Q_PROPERTY(float panSpeed READ panSpeed WRITE setPanSpeed NOTIFY panSpeedChanged)
    Q_PROPERTY(float zoomSpeed READ zoomSpeed WRITE setZoomSpeed NOTIFY zoomSpeedChanged)

    public:
        explicit CustomCameraController(Qt3DCore::QNode *parent = nullptr);
        ~CustomCameraController() override;

        Qt3DRender::QCamera *camera() const { return m_camera; }
        float rotationSpeed() const { return m_rotationSpeed; }
        float panSpeed() const { return m_panSpeed; }
        float zoomSpeed() const { return m_zoomSpeed; }

        void setCamera(Qt3DRender::QCamera *camera);
        void setRotationSpeed(float speed);
        void setPanSpeed(float speed);
        void setZoomSpeed(float speed);

        // Set the point to orbit around
        void setTarget(const QVector3D &target);
        QVector3D target() const { return m_target; }

        // Reset camera to default view
        void resetView();

    Q_SIGNALS:
        void cameraChanged();
        void rotationSpeedChanged();
        void panSpeedChanged();
        void zoomSpeedChanged();

    protected:
        bool eventFilter(QObject *obj, QEvent *event) override;

    private:
        void updateCameraPosition();

        Qt3DRender::QCamera *m_camera = nullptr;
        Qt3DInput::QMouseDevice *m_mouseDevice = nullptr;
        Qt3DInput::QMouseHandler *m_mouseHandler = nullptr;

        // Camera state
        QVector3D m_target{0.0f, 0.0f, 0.0f};  // Point to orbit around
        float m_distance = 10.0f;               // Distance from target
        float m_azimuth = 0.0f;                 // Horizontal angle (radians)
        float m_elevation = 0.5f;               // Vertical angle (radians), 0 = horizontal

        // Interaction state
        QPoint m_lastMousePos;
        bool m_leftButtonPressed = false;
        bool m_middleButtonPressed = false;
        bool m_rightButtonPressed = false;

        // Speeds
        float m_rotationSpeed = 0.005f;
        float m_panSpeed = 0.005f;
        float m_zoomSpeed = 0.1f;

        // Limits
        float m_minDistance = 0.5f;
        float m_maxDistance = 100.0f;
        float m_minElevation = -1.5f;  // Just above -90 degrees
        float m_maxElevation = 1.5f;   // Just below 90 degrees
};

#endif // CUSTOM_CAMERA_CONTROLLER_H