#ifndef ICP_H
#define ICP_H

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <opencv2/opencv.hpp>

class icp {
private:
    std::vector<Eigen::Vector2d> source_points_;
    std::vector<Eigen::Vector2d> target_points_;
    Eigen::Matrix2d R_;
    Eigen::Vector2d t_;

public:

    // Empty constructor
    icp(){};

    //Destructor
    ~icp(){};

    // Constru ctor
    icp(const std::vector<Eigen::Vector2d>& source_points, const std::vector<Eigen::Vector2d>& target_points);

    // set source points and target points
    void setPoints(const std::vector<Eigen::Vector2d>& source_points, const std::vector<Eigen::Vector2d>& target_points);

    // Función para obtener la matriz de rotación resultante
    Eigen::Matrix2d rotation() const;

    // Función para obtener el vector de traslación resultante
    Eigen::Vector2d translation() const;

    Eigen::Vector2d transformPoint(const Eigen::Vector2d& point) const;

    // Función pública para ejecutar el algoritmo ICP
    void align();

    // Función para dibujar los puntos fuente y destino en una ventana de OpenCV
    void drawPoints() const;
};

#endif // ICP_H
