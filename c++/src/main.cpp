

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <ostream>
#include <sys/time.h>
#include <regex>
#include <string>
#include <chrono>
#include <thread>
#include "edgetpu.h"
#include "opencv2/opencv.hpp"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

#include "detection_camera.h"
#include "tflite_wrapper.h"

#define DOCHECK(x)\
  if(!(x))\
  {\
  	fprintf(stderr,"ERROR  %s:%d\n",__FILE__,__LINE__);\
  	exit(1);\
  }\

int main(int argc, char** argv) 
{ 
  const bool with_edgetpu = true;
  const int image_width = CAMERA_WIDTH;
  const int image_height = CAMERA_HEIGHT;
  const bool verbose = true;
  const bool post_processing = true;
  const int videoSource = 0;

  // Get edgetpu context.
  std::cout << "seeking" << std::endl;
  std::vector<edgetpu::EdgeTpuManager::DeviceEnumerationRecord> records = edgetpu::EdgeTpuManager::GetSingleton()->EnumerateEdgeTpu();
  DOCHECK(records.size() > 0);
  for(const auto& value: records)
  { std::cout << "found at " << value.path << std::endl; } 

  //std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->NewEdgeTpuContext();
  std::shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
  DOCHECK(edgetpu_context != nullptr);
  std::cout << "opened" << std::endl;
  DOCHECK(edgetpu_context.get() != nullptr);
  DetectionCamera dc("../model/face_detection_full_integer_edgetpu.tflite", 
      edgetpu_context.get(), with_edgetpu, videoSource, image_height,
      image_width, verbose, post_processing);
  std::cout << "initialized!\n";  
  dc.Run();
  return 0;
}
