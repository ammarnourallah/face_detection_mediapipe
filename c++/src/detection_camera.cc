#include "detection_camera.h"
#include "image_helpers.h"
#include "edgetpu.h"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include <chrono>
#include <omp.h>
#include <math.h>
#include "opencv2/imgproc/imgproc.hpp"

DetectionCamera::DetectionCamera(
    const std::string& detect_model, 
    edgetpu::EdgeTpuContext* edgetpu_context, const bool edgetpu, const int source,
    const int height, const int width, const bool verbose, const bool post_processing)
    : m_wrapper(detect_model, edgetpu_context, edgetpu),      
      m_camera(source),
      m_height(height),
      m_width(width),
      m_verbose(verbose)
      {
      }

DetectionCamera::~DetectionCamera() {

}
  
void DetectionCamera::DebugLog(const char* log)
{
  if(m_verbose)
  {
    std::cout << log << std::endl;
  }
}

inline cv::Point2f getRotationPoint(float x, float y, float cx, float cy, float co, float si)
{
  float x1 = (x - cx) * co + (y - cy) * si + cx;
  float y1 = -(x - cx) * si + (y - cy) * co + cy;
  return cv::Point2f(x1, y1);
}

cv::Mat DetectionCamera::NormalizationfromRect(cv::Mat image, sBoxInfo *pBox, int len, float scale, cv::Point2f *pRotPoints, cv::Mat& transform) {
  float dy = pBox->lPoint.y - pBox->rPoint.y;
  float dx = pBox->rPoint.x - pBox->lPoint.x;
  float angle = atan2(dy, dx);
  float sx, sy, ex, ey, co, si, length;
  cv::Point2f pNormPoints[4];
  std::cout << angle << std::endl;
  co = cos(angle);
  si = sin(angle);
  length = pBox->sRect.width * scale / 2;
  sx = pBox->sRect.cx - length;
  ex = pBox->sRect.cx + length;
  sy = pBox->sRect.cy - length;
  ey = pBox->sRect.cy + length;

  pRotPoints[0] = getRotationPoint(sx, sy, pBox->sRect.cx, pBox->sRect.cy, co, si);
  pRotPoints[1] = getRotationPoint(ex, sy, pBox->sRect.cx, pBox->sRect.cy, co, si);
  pRotPoints[2] = getRotationPoint(ex, ey, pBox->sRect.cx, pBox->sRect.cy, co, si);
  pRotPoints[3] = getRotationPoint(sx, ey, pBox->sRect.cx, pBox->sRect.cy, co, si);

  pNormPoints[0] = cv::Point2f(0, 0);
  pNormPoints[1] = cv::Point2f(len, 0);
  pNormPoints[2] = cv::Point2f(len, len);
  pNormPoints[3] = cv::Point2f(0, len);
  transform = cv::getPerspectiveTransform(pRotPoints, pNormPoints);

  cv::Mat output(cv::Size(len, len), CV_32FC3);
  cv::warpPerspective(image, output, transform, output.size());
  transform = cv::getPerspectiveTransform(pNormPoints, pRotPoints);

  return output;
}

void DetectionCamera::Run() {
  std::cout << "Starting detection camera\n"; 
  // Initializing cameras.
  if (!m_camera.isOpened()) 
  {
    std::cerr << "Unable to open camera!\n";
    exit(0);
  } 
  else 
  {
    std::cout << "Camera Started\n"; 
    m_camera.set(cv::CAP_PROP_FPS, 20.0f);
    m_camera.set(cv::CAP_PROP_FRAME_HEIGHT, m_height);
    m_camera.set(cv::CAP_PROP_FRAME_WIDTH, m_width);
    int codec = cv::VideoWriter::fourcc('B','G','R','3');
    m_camera.set(cv::CAP_PROP_FOURCC, codec);
    printf("\t-Set camera buffer size\r\n");
    m_camera.set(cv::CAP_PROP_BUFFERSIZE, 1);
    // In case user gives incorrect parameters, cv will re-adjust, we reset our
    // values to fit cv.
    m_height = m_camera.get(cv::CAP_PROP_FRAME_HEIGHT);
    m_width = m_camera.get(cv::CAP_PROP_FRAME_WIDTH);
    std::cout << m_width << "x" << m_height << " Camera Configured\n";
  }

  const auto& cvblue = cv::Scalar(255, 0, 0);
  const auto& cvgreen = cv::Scalar(0, 255, 0);
  const auto& cvred = cv::Scalar(0, 0, 255);

  cv::Mat frame, frame_pad, frame_resized, inputs(cv::Size(128, 128), CV_32FC3, m_wrapper.GetDetectInputBuffer());

  double total_duration = 0;
  bool print_time = true;
  m_camera.read(frame);
  frame_pad.create(m_width, m_width, frame.type());
  frame_pad.setTo(cv::Scalar::all(0));
  int padding = (m_width - m_height) / 2;
  const auto& rect = cv::Rect(0, padding, m_width, m_height);

  // landmark transform infos
  cv::Point2f pRotPoints[4]; 
  cv::Mat transform;
  cv::Mat normImage;

  std::cout << "Begin main loop\n"; 
  while (1) 
  { 
    if (!m_camera.read(frame)) {break;}  // Blank frame!
    ++m_frame_counter;

    auto st0 = std::chrono::system_clock::now();

    // padding
    frame.copyTo(frame_pad(rect));
    cv::resize(frame_pad, frame_resized, inputs.size());
    cv::cvtColor(frame_resized, frame_resized,cv::COLOR_BGR2RGB);
    frame_resized.convertTo(inputs, CV_32FC3, 1/127.5, -1);

    auto st1 = std::chrono::system_clock::now();
    bool res = m_wrapper.RunDetectInference(&m_sFaceInfo);
    if (res) {
      // rescale face rect according to image size
      m_sFaceInfo.sRect.cx *= m_width;
      m_sFaceInfo.sRect.cy = m_sFaceInfo.sRect.cy * m_width - padding;
      m_sFaceInfo.sRect.width *= m_width;
      m_sFaceInfo.lPoint.x *= m_width;
      m_sFaceInfo.lPoint.y = m_sFaceInfo.lPoint.y * m_width - padding;
      m_sFaceInfo.rPoint.x *= m_width;
      m_sFaceInfo.rPoint.y = m_sFaceInfo.rPoint.y * m_width - padding;

      normImage = NormalizationfromRect(frame, &m_sFaceInfo, m_nLandImageLen, m_flLandScale, pRotPoints, transform);
    }
    auto st2 = std::chrono::system_clock::now();

    std::chrono::microseconds duration;
    if (print_time) {
      duration = std::chrono::duration_cast<std::chrono::microseconds>(st1 - st0);
      std::cout << "Preprocessing Time:: " << duration.count()/1000.0  << " milliseconds\n";
      duration = std::chrono::duration_cast<std::chrono::microseconds>(st2 - st1);
      std::cout << "Inference Time:: " << duration.count()/1000.0  << " milliseconds\n";
      duration = std::chrono::duration_cast<std::chrono::microseconds>(st2 - st0);
      std::cout << "Total Processing Time: " << duration.count() /1000.0 << " milliseconds\n\n";
    }
    
    duration = std::chrono::duration_cast<std::chrono::microseconds>(st2 - st0);
    total_duration += duration.count();
    int average_dration = int(total_duration / m_frame_counter);
    const auto& f = "Inference Rate: "
                    + std::to_string(1000000 / average_dration) + " fps";
    std::cout << std::to_string(1000000 / average_dration) + " fps" << std::endl;
    cv::putText(frame, f, cv::Point(0, 20), cv::FONT_HERSHEY_COMPLEX, .8, cvblue, 1.5, 8, 0);

    if (res) {
      // draw rectangle
      int sx = int(m_sFaceInfo.sRect.cx - m_sFaceInfo.sRect.width / 2);
      int ex = int(m_sFaceInfo.sRect.cx + m_sFaceInfo.sRect.width / 2);
      int sy = int(m_sFaceInfo.sRect.cy - m_sFaceInfo.sRect.width / 2);
      int ey = int(m_sFaceInfo.sRect.cy + m_sFaceInfo.sRect.width / 2);
      cv::rectangle(frame, cv::Point(sx, sy), cv::Point(ex, ey), cvblue);

      // draw Point
      int x, y;
      x = int(m_sFaceInfo.lPoint.x);
      y = int(m_sFaceInfo.lPoint.y);
      cv::circle(frame, cv::Point(x, y), 2, cvgreen, cv::FILLED);
      x = int(m_sFaceInfo.rPoint.x);
      y = int(m_sFaceInfo.rPoint.y);
      cv::circle(frame, cv::Point(x, y), 2, cvgreen, cv::FILLED);

      // draw rotate rect
      cv::line(frame, cv::Point(int(pRotPoints[0].x), int(pRotPoints[0].y)), cv::Point(int(pRotPoints[1].x), int(pRotPoints[1].y)), cvred);
      cv::line(frame, cv::Point(int(pRotPoints[1].x), int(pRotPoints[1].y)), cv::Point(int(pRotPoints[2].x), int(pRotPoints[2].y)), cvred);
      cv::line(frame, cv::Point(int(pRotPoints[2].x), int(pRotPoints[2].y)), cv::Point(int(pRotPoints[3].x), int(pRotPoints[3].y)), cvred);
      cv::line(frame, cv::Point(int(pRotPoints[3].x), int(pRotPoints[3].y)), cv::Point(int(pRotPoints[0].x), int(pRotPoints[0].y)), cvred);
    }
    cv::imshow("Live Inference", frame);
    cv::waitKey(1);
  }
}

