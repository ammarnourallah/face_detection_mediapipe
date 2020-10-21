#ifndef EDGETPU_TFLITE_CV_DETECTION_CAMERA_H_
#define EDGETPU_TFLITE_CV_DETECTION_CAMERA_H_

#include "edgetpu.h"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tflite_wrapper.h"

#define CAMERA_WIDTH    640
#define CAMERA_HEIGHT   480
#define PI      3.1415926535897932384626433832795
class DetectionCamera {
public:
  // Constructors.
  DetectionCamera(
      const std::string& detect_model,
      edgetpu::EdgeTpuContext* edgetpu_context, const bool edgetpu,
      const int source, const int height, const int width, const bool verbose, const bool post_processing);
  DetectionCamera() = delete;
  // Loops camera frame and performs inference on each frame.
  void Run();
  void DebugLog(const char* log);
  // Destructor.
  ~DetectionCamera();

private:
  cv::Mat NormalizationfromRect(cv::Mat image, sBoxInfo *pBox, int len, float scale, cv::Point2f *pRotPoints, cv::Mat& transform);

  TfLiteWrapper m_wrapper;
  cv::VideoCapture m_camera;
  int m_height = 0;
  int m_width = 0;
  int m_nLandImageLen = 192;
  float m_flLandScale = 1.5;
  
  bool m_verbose = false;
  size_t m_frame_counter{0};
  sBoxInfo m_sFaceInfo;

};

#endif  // EDGETPU_TFLITE_CV_TFLITE_WRAPPER_H
