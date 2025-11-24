#include "yolo_detector.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


YOLODetector::YOLODetector(const std::string& modelPath,
                           const std::vector<std::string>& classNames,
                           float confThreshold,
                           float iouThreshold,
                           int inputSize,
                           bool useGPU)
    : m_device(torch::kCPU)
    , m_inputSize(inputSize)
    , m_confThreshold(confThreshold)
    , m_iouThreshold(iouThreshold)
    , m_useGPU(useGPU)
{
    try {
        // Load the model
        m_model = torch::jit::load(modelPath);
        
        // Set device
        if (useGPU && torch::cuda::is_available())
        {
            m_device = torch::Device(torch::kCUDA);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            m_device = torch::Device(torch::kCPU);
            std::cout << "Using CPU device" << std::endl;
        }
        
        m_model.to(m_device);
        m_model.eval();
        
        // Load class names
        if (classNames.empty())
            m_classNames = getDefaultClassNames();
         else
            m_classNames = classNames;

        std::cout << "YOLODetector initialized successfully" << std::endl;
        std::cout << "Model: " << modelPath << std::endl;
        std::cout << "Classes: " << m_classNames.size() << std::endl;
        std::cout << "Input size: " << m_inputSize << std::endl;
        
    } catch (const c10::Error& e) { throw std::runtime_error("Error loading model: " + std::string(e.what())); }
}

YOLODetector::~YOLODetector() {
    // Cleanup if needed
}

cv::Mat YOLODetector::captureImage()
{
    // TO BE IMPLEMENTED BY USER
    // This is a placeholder method
    // User should implement their own image acquisition logic here
    // Example:
    // cv::VideoCapture cap(0);
    // cv::Mat frame;
    // cap >> frame;
    // return frame;
    
    std::cerr << "Warning: captureImage() not implemented!" << std::endl;
    return cv::Mat();
}

std::vector<YOLODetector::Detection> YOLODetector::detect_doors(const cv::Mat& image)
{
    if (image.empty()) {
        std::cerr << "Empty image provided to detect()" << std::endl;
        return {};
    }
    
    cv::Size originalSize = image.size();
    
    // Preprocess image
    torch::Tensor inputTensor = preprocessImage(image);

    // Run inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(inputTensor);

    torch::NoGradGuard no_grad;
    auto output = m_model.forward(inputs).toTensor();
    std::cout << "1" << std::endl;

    // Postprocess output
    return postprocessOutput(output, originalSize);
}

std::vector<YOLODetector::Detection> YOLODetector::detectFromCapture()
{
    cv::Mat image = captureImage();
    if (image.empty()) {
        std::cerr << "Failed to capture image" << std::endl;
        return {};
    }
    return detect_doors(image);
}

torch::Tensor YOLODetector::preprocessImage(const cv::Mat& image) {
    cv::Mat rgbImage, resizedImage, floatImage;
    
    // Convert BGR to RGB
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
    
    // Resize with letterbox (maintain aspect ratio)
    int origWidth = image.cols;
    int origHeight = image.rows;
    
    float scale = std::min(
        static_cast<float>(m_inputSize) / origWidth,
        static_cast<float>(m_inputSize) / origHeight
    );
    
    int newWidth = static_cast<int>(origWidth * scale);
    int newHeight = static_cast<int>(origHeight * scale);
    
    cv::resize(rgbImage, resizedImage, cv::Size(newWidth, newHeight));
    
    // Create letterbox (padding)
    cv::Mat letterboxImage = cv::Mat::zeros(m_inputSize, m_inputSize, CV_8UC3);
    int topPad = (m_inputSize - newHeight) / 2;
    int leftPad = (m_inputSize - newWidth) / 2;
    
    resizedImage.copyTo(letterboxImage(cv::Rect(leftPad, topPad, newWidth, newHeight)));
    
    // Convert to float and normalize [0, 1]
    letterboxImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
    
    // Convert to tensor and change layout from HWC to CHW
    torch::Tensor tensor = torch::from_blob(
        floatImage.data,
        {1, m_inputSize, m_inputSize, 3},
        torch::kFloat32
    ).permute({0, 3, 1, 2}).contiguous();
    
    return tensor.to(m_device);
}

std::vector<YOLODetector::Detection> YOLODetector::postprocessOutput(const torch::Tensor& output,
                                                       const cv::Size& originalSize) {
    std::vector<Detection> detections;
    
    // YOLOv11 output format: [batch, num_detections, 84]
    // where 84 = 4 (bbox) + 80 (classes for COCO)
    auto outputAccessor = output.accessor<float, 3>();
    
    int numDetections = output.size(1);
    int numClasses = output.size(2) - 4;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<size_t> classIds;
    
    // Calculate scale factor for coordinate conversion
    float scale = std::min(
        static_cast<float>(m_inputSize) / originalSize.width,
        static_cast<float>(m_inputSize) / originalSize.height
    );
    
    int leftPad = (m_inputSize - static_cast<int>(originalSize.width * scale)) / 2;
    int topPad = (m_inputSize - static_cast<int>(originalSize.height * scale)) / 2;
    
    // Parse detections
    for (int i = 0; i < numDetections; i++) {
        // Get class scores
        float maxScore = 0.0f;
        int maxClassId = 0;
        
        for (int j = 0; j < numClasses; j++) {
            float score = outputAccessor[0][i][4 + j];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = j;
            }
        }
        
        // Filter by confidence threshold
        if (maxScore < m_confThreshold) {
            continue;
        }
        
        // Get bbox coordinates (center_x, center_y, width, height)
        float cx = outputAccessor[0][i][0];
        float cy = outputAccessor[0][i][1];
        float w = outputAccessor[0][i][2];
        float h = outputAccessor[0][i][3];
        
        // Convert to corner coordinates and scale back to original size
        int x1 = static_cast<int>((cx - w / 2 - leftPad) / scale);
        int y1 = static_cast<int>((cy - h / 2 - topPad) / scale);
        int x2 = static_cast<int>((cx + w / 2 - leftPad) / scale);
        int y2 = static_cast<int>((cy + h / 2 - topPad) / scale);
        
        // Clamp to image bounds
        x1 = std::max(0, std::min(x1, originalSize.width - 1));
        y1 = std::max(0, std::min(y1, originalSize.height - 1));
        x2 = std::max(0, std::min(x2, originalSize.width - 1));
        y2 = std::max(0, std::min(y2, originalSize.height - 1));
        
        cv::Rect box(x1, y1, x2 - x1, y2 - y1);
        
        boxes.push_back(box);
        scores.push_back(maxScore);
        classIds.push_back(maxClassId);
    }
    
    // Apply Non-Maximum Suppression
    std::vector<int> indices = nms(boxes, scores, m_iouThreshold);
    
    // Create final detections
    for (int idx : indices) {
        std::string label = (classIds[idx] < m_classNames.size())
                          ? m_classNames[classIds[idx]]
                          : "class_" + std::to_string(classIds[idx]);

        detections.emplace_back(boxes[idx], classIds[idx], label, scores[idx]);
    }

    return detections;
}

std::vector<int> YOLODetector::nms(const std::vector<cv::Rect>& boxes,
                                   const std::vector<float>& scores,
                                   float iouThreshold) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    // Sort by score (descending)
    std::sort(indices.begin(), indices.end(), [&scores](int i1, int i2) {
        return scores[i1] > scores[i2];
    });
    
    std::vector<int> keep;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        if (suppressed[idx]) continue;
        
        keep.push_back(idx);
        
        for (size_t j = i + 1; j < indices.size(); j++) {
            int idx2 = indices[j];
            if (suppressed[idx2]) continue;
            
            float iou = calculateIoU(boxes[idx], boxes[idx2]);
            if (iou > iouThreshold) {
                suppressed[idx2] = true;
            }
        }
    }
    
    return keep;
}

float YOLODetector::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    
    int intersectionWidth = std::max(0, x2 - x1);
    int intersectionHeight = std::max(0, y2 - y1);
    int intersectionArea = intersectionWidth * intersectionHeight;
    
    int box1Area = box1.width * box1.height;
    int box2Area = box2.width * box2.height;
    int unionArea = box1Area + box2Area - intersectionArea;
    
    return (unionArea > 0) ? static_cast<float>(intersectionArea) / unionArea : 0.0f;
}

std::vector<std::string> YOLODetector::getDefaultClassNames() {
    // COCO dataset class names
    return {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };
}
