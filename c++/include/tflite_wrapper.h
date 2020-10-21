#ifndef EDGETPU_TFLITE_CV_TFLITE_WRAPPER_H_
#define EDGETPU_TFLITE_CV_TFLITE_WRAPPER_H_

#include <array>
#include <chrono>
#include <map>
#include <string>
#include "opencv2/opencv.hpp"

#include "edgetpu.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/kernels/register.h"

struct sRectInfo
{
  float cx, cy;
  float width;
};

struct sBoxInfo
{
  sRectInfo sRect;
  cv::Point2f lPoint, rPoint;
};

class TfLiteWrapper {
public:
  // Constructors.
  TfLiteWrapper(
      const std::string& detect_model_path, 
      edgetpu::EdgeTpuContext* edgetpu_context, const bool edgetpu);
  TfLiteWrapper() = delete;
  // Initializes a tflite::Interpreter for CPU usage.
  void InitTfLiteWrapper();
  // Initializes a tflite::Interpreter with edgetpu custom ops.
  void InitTfLiteWrapperEdgetpu(edgetpu::EdgeTpuContext* edgetpu_context);
  // Runinference;
  bool RunDetectInference(sBoxInfo *pFace);
  float *GetDetectInputBuffer() {
    return m_pDetectInput;
  };
  int GetTensorSize() {
    return m_image_size;
  }
  // Destructor.
  ~TfLiteWrapper() = default;

private:
  std::unique_ptr<tflite::Interpreter> detect_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> detect_model;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  int m_image_size = 128 * 128 * 3;
  int m_nFaceRectSize = 896;
  float *m_pDetectInput;
  float m_flFaceThreshold = 1.f;
  float *m_pflRawScore, *m_pflRawBox;
  float *m_pflAnchors;
};

#endif  // EDGETPU_TFLITE_CV_TFLITE_WRAPPER_H
