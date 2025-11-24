#ifndef YOLO_DETECTOR_ONNX_H
#define YOLO_DETECTOR_ONNX_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

struct Detection {
    cv::Rect roi;
    int classId;
    std::string label;
    float score;
    
    Detection(const cv::Rect& r, int id, const std::string& lbl, float s)
        : roi(r), classId(id), label(lbl), score(s) {}
};

class YOLODetectorONNX {
public:
    YOLODetectorONNX(const std::string& modelPath,
                     const std::vector<std::string>& classNames = {},
                     float confThreshold = 0.25f,
                     float iouThreshold = 0.45f,
                     int inputSize = 640,
                     bool useGPU = true);
    
    ~YOLODetectorONNX();
    
    cv::Mat captureImage();
    std::vector<Detection> detect(const cv::Mat& image);
    std::vector<Detection> detectFromCapture();
    
    void setConfThreshold(float threshold) { m_confThreshold = threshold; }
    void setIouThreshold(float threshold) { m_iouThreshold = threshold; }
    
private:
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::SessionOptions m_sessionOptions;
    
    std::vector<std::string> m_classNames;
    std::vector<const char*> m_inputNames;
    std::vector<const char*> m_outputNames;
    
    int m_inputSize;
    float m_confThreshold;
    float m_iouThreshold;
    
    std::vector<float> preprocessImage(const cv::Mat& image);
    std::vector<Detection> postprocessOutput(const std::vector<float>& output,
                                             const std::vector<int64_t>& outputShape,
                                             const cv::Size& originalSize);
    
    std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                         const std::vector<float>& scores,
                         float iouThreshold);
    
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    std::vector<std::string> getDefaultClassNames();
};

#endif // YOLO_DETECTOR_ONNX_H
