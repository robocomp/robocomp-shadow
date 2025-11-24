#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

class YOLODetector
{
    public:

        struct Detection
        {
            cv::Rect roi;           // Bounding box (x, y, width, height)
            int classId;            // Class ID
            std::string label;      // Class label/name
            float score;            // Confidence score

            Detection(const cv::Rect& r, int id, const std::string& lbl, float s)
                : roi(r), classId(id), label(lbl), score(s) {}
        };

        /**
         * Constructor
         * @param modelPath Path to the TorchScript model file
         * @param classNames Vector of class names (COCO classes by default)
         * @param confThreshold Confidence threshold for detections (default: 0.25)
         * @param iouThreshold IoU threshold for NMS (default: 0.45)
         * @param inputSize Input size for the model (default: 640)
         * @param useGPU Whether to use GPU for inference (default: true)
         */
        YOLODetector(const std::string& modelPath,
                     const std::vector<std::string>& classNames = {},
                     float confThreshold = 0.25f,
                     float iouThreshold = 0.45f,
                     int inputSize = 640,
                     bool useGPU = true);

        ~YOLODetector();

        /**
         * Capture image from camera/sensor
         * @return cv::Mat containing the captured image
         */
        cv::Mat captureImage();

        /**
         * Run inference on an image
         * @param image Input image (BGR format)
         * @return Vector of detections with ROI, label, and score
         */
        std::vector<Detection> detect_doors(const cv::Mat& image);

        /**
         * Run inference on the latest captured image
         * @return Vector of detections with ROI, label, and score
         */
        std::vector<Detection> detectFromCapture();

        // Getters/Setters
        void setConfThreshold(float threshold) { m_confThreshold = threshold; }
        void setIouThreshold(float threshold) { m_iouThreshold = threshold; }
        float getConfThreshold() const { return m_confThreshold; }
        float getIouThreshold() const { return m_iouThreshold; }

    private:
        torch::jit::script::Module m_model;
        torch::Device m_device;
        std::vector<std::string> m_classNames;

        int m_inputSize;
        float m_confThreshold;
        float m_iouThreshold;
        bool m_useGPU;

        /**
         * Preprocess image for YOLO input
         * @param image Input image (BGR)
         * @return Preprocessed tensor ready for inference
         */
        torch::Tensor preprocessImage(const cv::Mat& image);

        /**
         * Postprocess model output to extract detections
         * @param output Model output tensor
         * @param originalSize Original image size for scaling coordinates
         * @return Vector of detections after NMS
         */
        std::vector<Detection> postprocessOutput(const torch::Tensor& output,
                                                 const cv::Size& originalSize);

        /**
         * Non-Maximum Suppression
         * @param boxes Vector of bounding boxes
         * @param scores Vector of confidence scores
         * @param iouThreshold IoU threshold
         * @return Indices of boxes to keep
         */
        std::vector<int> nms(const std::vector<cv::Rect>& boxes,
                             const std::vector<float>& scores,
                             float iouThreshold);

        /**
         * Calculate IoU between two boxes
         */
        float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);

        /**
         * Load default COCO class names
         */
        std::vector<std::string> getDefaultClassNames();
};

#endif // YOLO_DETECTOR_H
