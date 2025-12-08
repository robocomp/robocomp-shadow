#pragma once

// --- Librerías de terceros pesadas y estables ---
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <boost/circular_buffer.hpp>

// --- Cabeceras de Qt ---
#include <QDebug>
#include <QWidget>
#include <QGraphicsScene>
#include <QLabel>
#include <QMainWindow>
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

// --- Cabeceras del proyecto que son grandes y estables ---
#include "qcustomplot.h"
