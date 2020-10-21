#include "tflite_wrapper.h"

#include <time.h>
#include <iostream>
#include <memory>
#include <vector>
#include "opencv2/opencv.hpp"
#include "const.h"

#define DOCHECK(x)\
if(!(x))\
{\
  fprintf(stderr,"ERROR  %s:%d\n",__FILE__,__LINE__);\
  exit(1);\
}\


TfLiteWrapper::TfLiteWrapper(
    const std::string& detect_model_path,
    edgetpu::EdgeTpuContext* edgetpu_context, const bool edgetpu) 
{
  detect_model = tflite::FlatBufferModel::BuildFromFile(detect_model_path.c_str());
  DOCHECK(detect_model != nullptr);
  // Initializes interpreter.
  
  if (edgetpu && edgetpu_context) 
  {
    std::cout << "acceleration enabled\n";
    InitTfLiteWrapperEdgetpu(edgetpu_context);
  } 
  else 
  {
    std::cout << "acceleration disabled\n";
    InitTfLiteWrapper();
  }
  detect_interpreter->SetNumThreads(1);
  if (detect_interpreter->AllocateTensors() != kTfLiteOk) 
  {
    std::cout << "Failed to allocate tensors of seg interpreter.\n";
  }
  
  m_pDetectInput = detect_interpreter->typed_input_tensor<float>(0);
  m_pflRawScore = detect_interpreter->typed_output_tensor<float>(0);
  m_pflRawBox = detect_interpreter->typed_output_tensor<float>(1);

  m_pflAnchors = (float *) g_pdwAnchors;
  std::cout << "Successfully Initialized.\n";
}

void TfLiteWrapper::InitTfLiteWrapperEdgetpu(edgetpu::EdgeTpuContext* edgetpu_context) 
{  
  resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());
  std::cout << "edgetpu::RegisterCustomOp()\n";
  if (tflite::InterpreterBuilder(*detect_model, resolver)(&detect_interpreter) != kTfLiteOk) {
    std::cout << "Failed to build Interpreter\n";
    std::abort();
  }
  detect_interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);
  std::cout << "Edge Interpreter built\n";

}

void TfLiteWrapper::InitTfLiteWrapper() {
  if (tflite::InterpreterBuilder(*detect_model, resolver)(&detect_interpreter) != kTfLiteOk) {
    std::cout << "Failed to build Interpreter\n";
    std::abort();
  }
}

bool TfLiteWrapper::RunDetectInference(sBoxInfo *pFace) {
  // std::cout << "Running Inference\n";  
  detect_interpreter->Invoke();

  // get maximum candidate
  float flMaxScore = m_flFaceThreshold;
  int nMaxIdx = 0;
  float *pflScore = m_pflRawScore;
  for (int i = 0; i < m_nFaceRectSize; i++, pflScore++)
  {
    if (flMaxScore < *pflScore) {
      flMaxScore = *pflScore;
      nMaxIdx = i;
    }
  }
  if (flMaxScore < m_flFaceThreshold)
    return false;

  // maximum confidence face 
  int index = nMaxIdx * 16;
  pFace->sRect.cx = m_pflRawBox[index] / 128 + m_pflAnchors[nMaxIdx * 2];
  pFace->sRect.cy = m_pflRawBox[index+1] / 128 + m_pflAnchors[nMaxIdx * 2 + 1];
  pFace->sRect.width = m_pflRawBox[index+2] / 128;
  pFace->lPoint.x = m_pflRawBox[index+4] / 128 + m_pflAnchors[nMaxIdx * 2];
  pFace->lPoint.y = m_pflRawBox[index+5] / 128 + m_pflAnchors[nMaxIdx * 2 + 1];
  pFace->rPoint.x = m_pflRawBox[index+6] / 128 + m_pflAnchors[nMaxIdx * 2];
  pFace->rPoint.y = m_pflRawBox[index+7] / 128 + m_pflAnchors[nMaxIdx * 2 + 1];
  return true;

  // check overlay decision
  pflScore = m_pflRawScore;
  int nFaceCnt = 0;
  float cx, cy, width, lx, ly, rx, ry, x, y, wid, sx, sy, ex, ey, area, max_area, min_area;
  max_area = pFace->sRect.width * pFace->sRect.width * 4;
  cx = cy = width = lx = ly = rx = ry = 0;
  index = 0;
  for (int i = 0; i < m_nFaceRectSize; i++, pflScore++, index += 16)
  {
    if (m_flFaceThreshold >= *pflScore) 
      continue;
    float a1, a2;
    a1 = m_pflAnchors[i * 2];
    a2 = m_pflAnchors[i * 2 + 1];
    x = m_pflRawBox[index] / 128 + a1;
    y = m_pflRawBox[index+1] / 128 + a2;
    wid = m_pflRawBox[index+2] / 128;

    sx = std::max(pFace->sRect.cx - pFace->sRect.width, x - wid);
    ex = std::min(pFace->sRect.cx + pFace->sRect.width, x + wid);
    sy = std::max(pFace->sRect.cy - pFace->sRect.width, y - wid);
    ey = std::min(pFace->sRect.cy + pFace->sRect.width, y + wid);
    area = (ex - sx) * (ey - sy);
    min_area = std::min(max_area, wid * wid * 4);
    if (area > min_area * 0.3) {
      cx += x;
      cy += y;
      width += wid;
      lx += m_pflRawBox[index+4] / 128 + a1;
      ly += m_pflRawBox[index+5] / 128 + a2;
      rx += m_pflRawBox[index+6] / 128 + a1;
      ry += m_pflRawBox[index+7] / 128 + a2;
      nFaceCnt++;
    }
  }

  pFace->sRect.cx = cx / nFaceCnt;
  pFace->sRect.cy = cy / nFaceCnt;
  pFace->sRect.width = width / nFaceCnt;
  pFace->lPoint.x = lx / nFaceCnt;
  pFace->lPoint.y = ly / nFaceCnt;
  pFace->rPoint.x = rx / nFaceCnt;
  pFace->rPoint.y = ry / nFaceCnt;
  return true;
}
