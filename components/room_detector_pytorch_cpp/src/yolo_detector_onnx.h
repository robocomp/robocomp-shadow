#ifndef YOLO_DETECTOR_ONNX_H
#define YOLO_DETECTOR_ONNX_H

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

struct Detection
{
    cv::Rect roi;
    int classId;
    std::string label;
    float score;

    Detection(const cv::Rect &r, int id, const std::string &lbl, float s)
        : roi(r), classId(id), label(lbl), score(s)
    {
    }
};

// Configuration for panoramic image slicing
struct SliceConfig
{
    int targetWidth = 640; // Target slice width (YOLO training size)
    int targetHeight = 480; // Target slice height
    float overlapRatio = 0.25f; // Overlap between adjacent slices (25%)
    float mergeIouThreshold = 0.5f; // IoU threshold for merging detections across slices

    SliceConfig() = default;

    SliceConfig(int w, int h, float overlap = 0.25f, float mergeIou = 0.5f)
        : targetWidth(w), targetHeight(h), overlapRatio(overlap), mergeIouThreshold(mergeIou)
    {
    }
};

class YOLODetectorONNX
{
public:
    YOLODetectorONNX(const std::string &modelPath,
                     const std::vector<std::string> &classNames = {},
                     float confThreshold = 0.25f,
                     float iouThreshold = 0.45f,
                     int inputSize = 640,
                     bool useGPU = true);

    ~YOLODetectorONNX();

    cv::Mat captureImage();

    // Original single-image detection
    std::vector<Detection> detect(const cv::Mat &image);

    std::vector<Detection> detectFromCapture();

    // Panoramic detection with automatic slicing
    std::vector<Detection> detectPanoramic(const cv::Mat &image, const SliceConfig &config = SliceConfig());

    // Batch detection on multiple images (useful for processing slices)
    std::vector<std::vector<Detection> > detectBatch(const std::vector<cv::Mat> &images);

    void setConfThreshold(float threshold) { m_confThreshold = threshold; }
    void setIouThreshold(float threshold) { m_iouThreshold = threshold; }

    // Get slice information for debugging/visualization
    struct SliceInfo
    {
        cv::Rect region; // Region in original image
        int sliceIndex;
    };

    std::vector<SliceInfo> getLastSliceInfo() const { return m_lastSliceInfo; }

private:
    std::unique_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::SessionOptions m_sessionOptions;

    std::vector<std::string> m_classNames;
    std::vector<const char *> m_inputNames;
    std::vector<const char *> m_outputNames;

    int m_inputSize;
    float m_confThreshold;
    float m_iouThreshold;
    int m_activeClassChannel;

    // Store slice info from last panoramic detection
    std::vector<SliceInfo> m_lastSliceInfo;

    std::vector<float> preprocessImage(const cv::Mat &image);

    std::vector<Detection> postprocessOutput(const std::vector<float> &output,
                                             const std::vector<int64_t> &outputShape,
                                             const cv::Size &originalSize);

    std::vector<int> nms(const std::vector<cv::Rect> &boxes,
                         const std::vector<float> &scores,
                         float iouThreshold);

    float calculateIoU(const cv::Rect &box1, const cv::Rect &box2);

    std::vector<std::string> getDefaultClassNames();

    // Panoramic slicing helpers
    std::vector<cv::Rect> computeSliceRegions(const cv::Size &imageSize, const SliceConfig &config);

    std::vector<Detection> mergeSliceDetections(const std::vector<std::vector<Detection> > &sliceDetections,
                                                const std::vector<cv::Rect> &sliceRegions,
                                                float mergeIouThreshold);
};

#endif // YOLO_DETECTOR_ONNX_H
