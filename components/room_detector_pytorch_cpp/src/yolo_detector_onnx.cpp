#include "yolo_detector_onnx.h"
#include <algorithm>
#include <numeric>
#include <iostream>

YOLODetectorONNX::YOLODetectorONNX(const std::string& modelPath,
                                   const std::vector<std::string>& classNames,
                                   float confThreshold,
                                   float iouThreshold,
                                   int inputSize,
                                   bool useGPU)
    : m_inputSize(inputSize)
    , m_confThreshold(confThreshold)
    , m_iouThreshold(iouThreshold)
{
    try {
        // Initialize ONNX Runtime
        m_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLODetector");
        
        m_sessionOptions.SetIntraOpNumThreads(1);
        m_sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        if (useGPU) {
            // Enable CUDA if available
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            m_sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
            std::cout << "Using CUDA" << std::endl;
        } else {
            std::cout << "Using CPU" << std::endl;
        }
        
        // Create session
#ifdef _WIN32
        std::wstring wideModelPath(modelPath.begin(), modelPath.end());
        m_session = std::make_unique<Ort::Session>(*m_env, wideModelPath.c_str(), m_sessionOptions);
#else
        m_session = std::make_unique<Ort::Session>(*m_env, modelPath.c_str(), m_sessionOptions);
#endif
        
        // Get input/output names
        Ort::AllocatorWithDefaultOptions allocator;
        
        size_t numInputNodes = m_session->GetInputCount();
        for (size_t i = 0; i < numInputNodes; i++) {
            auto inputName = m_session->GetInputNameAllocated(i, allocator);
            m_inputNames.push_back(strdup(inputName.get()));
        }
        
        size_t numOutputNodes = m_session->GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto outputName = m_session->GetOutputNameAllocated(i, allocator);
            m_outputNames.push_back(strdup(outputName.get()));
        }
        
        // Load class names
        if (classNames.empty()) {
            m_classNames = getDefaultClassNames();
        } else {
            m_classNames = classNames;
        }
        
        std::cout << "YOLODetectorONNX initialized successfully" << std::endl;
        std::cout << "Model: " << modelPath << std::endl;
        std::cout << "Input: " << m_inputNames[0] << std::endl;
        std::cout << "Output: " << m_outputNames[0] << std::endl;
        
    } catch (const Ort::Exception& e) {
        throw std::runtime_error("ONNX Runtime error: " + std::string(e.what()));
    }
}

YOLODetectorONNX::~YOLODetectorONNX() {
    for (auto name : m_inputNames) free(const_cast<char*>(name));
    for (auto name : m_outputNames) free(const_cast<char*>(name));
}

cv::Mat YOLODetectorONNX::captureImage() {
    std::cerr << "Warning: captureImage() not implemented!" << std::endl;
    return cv::Mat();
}

std::vector<Detection> YOLODetectorONNX::detect(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Empty image provided to detect()" << std::endl;
        return {};
    }
    
    cv::Size originalSize = image.size();
    
    // Preprocess
    std::vector<float> inputTensor = preprocessImage(image);
    
    // Create input tensor
    std::vector<int64_t> inputShape = {1, 3, m_inputSize, m_inputSize};
    
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensorOrt = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensor.data(), inputTensor.size(),
        inputShape.data(), inputShape.size()
    );
    
    // Run inference
    auto outputTensors = m_session->Run(
        Ort::RunOptions{nullptr},
        m_inputNames.data(), &inputTensorOrt, 1,
        m_outputNames.data(), 1
    );
    
    // Get output
    auto* outputData = outputTensors[0].GetTensorMutableData<float>();
    const auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    
    size_t outputSize = 1;
    for (const auto dim : outputShape) outputSize *= dim;

    const std::vector<float> output(outputData, outputData + outputSize);
    
    // Postprocess
    return postprocessOutput(output, outputShape, originalSize);
}

std::vector<Detection> YOLODetectorONNX::detectFromCapture() {
    cv::Mat image = captureImage();
    if (image.empty()) {
        std::cerr << "Failed to capture image" << std::endl;
        return {};
    }
    return detect(image);
}

std::vector<float> YOLODetectorONNX::preprocessImage(const cv::Mat& image) {
    cv::Mat rgbImage, resizedImage, floatImage;
    
    // BGR to RGB
    cv::cvtColor(image, rgbImage, cv::COLOR_BGR2RGB);
    
    // Calculate scale for letterbox
    float scale = std::min(
        static_cast<float>(m_inputSize) / image.cols,
        static_cast<float>(m_inputSize) / image.rows
    );
    
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);
    
    cv::resize(rgbImage, resizedImage, cv::Size(newWidth, newHeight));
    
    // Create letterbox with padding
    cv::Mat letterboxImage = cv::Mat::zeros(m_inputSize, m_inputSize, CV_8UC3);
    int topPad = (m_inputSize - newHeight) / 2;
    int leftPad = (m_inputSize - newWidth) / 2;
    
    resizedImage.copyTo(letterboxImage(cv::Rect(leftPad, topPad, newWidth, newHeight)));
    
    // Convert to float and normalize
    letterboxImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
    
    // Convert HWC to CHW format
    std::vector<float> inputTensor;
    inputTensor.resize(3 * m_inputSize * m_inputSize);
    
    std::vector<cv::Mat> channels(3);
    cv::split(floatImage, channels);
    
    for (int c = 0; c < 3; c++) {
        std::memcpy(
            inputTensor.data() + c * m_inputSize * m_inputSize,
            channels[c].data,
            m_inputSize * m_inputSize * sizeof(float)
        );
    }
    
    return inputTensor;
}

std::vector<Detection> YOLODetectorONNX::postprocessOutput(
    const std::vector<float>& output,
    const std::vector<int64_t>& outputShape,
    const cv::Size& originalSize)
{
    std::vector<Detection> detections;
    
    // YOLOv11 output: [1, 84, num_boxes] or [1, num_boxes, 84]
    int64_t numBoxes = outputShape[1];
    int64_t numClasses = outputShape[2] - 4;
    
    // Check if transpose is needed
    if (outputShape[2] > outputShape[1]) {
        numBoxes = outputShape[2];
        numClasses = outputShape[1] - 4;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    
    // Calculate scale
    float scale = std::min(
        static_cast<float>(m_inputSize) / originalSize.width,
        static_cast<float>(m_inputSize) / originalSize.height
    );
    
    int leftPad = (m_inputSize - static_cast<int>(originalSize.width * scale)) / 2;
    int topPad = (m_inputSize - static_cast<int>(originalSize.height * scale)) / 2;
    
    // Parse detections
    for (int64_t i = 0; i < numBoxes; i++) {
        // Get bbox and class scores
        int offset = outputShape[2] > outputShape[1] ? 
                    i * (numClasses + 4) : i;
        int stride = outputShape[2] > outputShape[1] ? 
                    1 : numBoxes;
        
        float cx = output[offset];
        float cy = output[offset + stride];
        float w = output[offset + 2 * stride];
        float h = output[offset + 3 * stride];
        
        // Find max class score
        float maxScore = 0.0f;
        int maxClassId = 0;
        
        for (int64_t j = 0; j < numClasses; j++) {
            float score = output[offset + (4 + j) * stride];
            if (score > maxScore) {
                maxScore = score;
                maxClassId = j;
            }
        }
        
        if (maxScore < m_confThreshold) continue;
        
        // Convert to corner coordinates
        int x1 = static_cast<int>((cx - w / 2 - leftPad) / scale);
        int y1 = static_cast<int>((cy - h / 2 - topPad) / scale);
        int x2 = static_cast<int>((cx + w / 2 - leftPad) / scale);
        int y2 = static_cast<int>((cy + h / 2 - topPad) / scale);
        
        // Clamp to image bounds
        x1 = std::max(0, std::min(x1, originalSize.width - 1));
        y1 = std::max(0, std::min(y1, originalSize.height - 1));
        x2 = std::max(0, std::min(x2, originalSize.width - 1));
        y2 = std::max(0, std::min(y2, originalSize.height - 1));
        
        boxes.emplace_back(x1, y1, x2 - x1, y2 - y1);
        scores.push_back(maxScore);
        classIds.push_back(maxClassId);
    }
    
    // Apply NMS
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

std::vector<int> YOLODetectorONNX::nms(const std::vector<cv::Rect>& boxes,
                                       const std::vector<float>& scores,
                                       float iouThreshold)
{
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    
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

float YOLODetectorONNX::calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
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

std::vector<std::string> YOLODetectorONNX::getDefaultClassNames() {
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
