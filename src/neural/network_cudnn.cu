/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#include <cassert>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include "neural/factory.h"
#include "utils/bititer.h"
#include "utils/exception.h"

#include <cublas_v2.h>
#include <cudnn.h>
#include <algorithm>

//#define DEBUG_RAW_NPS
// single kernel for entire SE operation (right now supported only for fp16)
// 
// (not ready yet, needs kernel to be compiled with SM7_0 (or higher). 
// need to refactor some fp16 specific code to make it work on all GPUs
//#define FUSED_SE_LAYER

namespace lczero {
namespace {

void CudnnError(cudnnStatus_t status, const char* file, const int& line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUDNN error: %s (%s:%d) ", cudnnGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

const char* CublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown cublas error";
}

void CublasError(cublasStatus_t status, const char* file, const int& line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUDNN error: %s (%s:%d) ", CublasGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

void CudaError(cudaError_t status, const char* file, const int& line) {
  if (status != cudaSuccess) {
    char message[128];
    sprintf(message, "CUDA error: %s (%s:%d) ", cudaGetErrorString(status),
            file, line);
    throw Exception(message);
  }
}

#define ReportCUDNNErrors(status) CudnnError(status, __FILE__, __LINE__)
#define ReportCUBLASErrors(status) CublasError(status, __FILE__, __LINE__)
#define ReportCUDAErrors(status) CudaError(status, __FILE__, __LINE__)

static constexpr int kNumOutputPolicy = 1858;

// The Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval.

template <typename DataType>
class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }

  BaseLayer(int c, int h, int w, BaseLayer* ip);
  size_t GetOutputSize(int N) const { return sizeof(DataType) * N * C * H * W; }

  // input2 is optional (skip connection).
  virtual void Eval(int N, DataType* output, const DataType* input,
                    const DataType* input2, void* scratch, size_t scratch_size,
                    cudnnHandle_t cudnn, cublasHandle_t cublas) = 0;

 protected:
  BaseLayer* input_;

  int C;  // Output tensor dimensions.
  int H;
  int W;
};

template <typename DataType>
class ConvLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;

 public:
  ConvLayer(BaseLayer<DataType>* ip, int C, int H, int W, int size, int Cin,
            bool relu = false, bool bias = false);
  ~ConvLayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas) override;

 private:
  const int c_input_;
  const int filter_size_;
  const bool use_relu_;
  const bool use_bias_;

  DataType* biases = nullptr;
  DataType* weights = nullptr;

  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t conv_algo_;

  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t in_tensor_desc_;
  cudnnTensorDescriptor_t out_tensor_desc_;
  cudnnActivationDescriptor_t activation_;
};

template <typename DataType>
class SoftMaxLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;

 public:
  SoftMaxLayer(BaseLayer<DataType>* ip);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas) override;

 private:
  cudnnTensorDescriptor_t out_tensor_desc_;
};

template <typename DataType>
class BNLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;

 public:
  BNLayer(BaseLayer<DataType>* ip, bool relu);
  ~BNLayer();

  void LoadWeights(float* cpuMeans, float* cpuVar);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas) override;

 private:
  const bool use_relu_;

  // Weights for BN layer are always in float irrespective of DataType
  // as there is not much point in converting these to fp16.
  float* means_ = nullptr;
  float* variances_ = nullptr;
};

template <typename DataType>
class FCLayer : public BaseLayer<DataType> {
 public:
  FCLayer(BaseLayer<DataType>* ip, int C, int H, int W, bool relu, bool bias,
          bool tanh = false, bool sigmoid = false);
  ~FCLayer();

  void LoadWeights(float* cpuWeight, float* cpuBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas) override;

 private:
  const bool use_bias_;
  const bool use_relu_;
  const bool use_tanh_;
  const bool use_sigmoid_;
  DataType* weights_ = nullptr;
  DataType* biases_ = nullptr;
};

// fused SE layer
// (optional bias add +) global avg -> FC1 -> FC2 -> global scale -> add skip
// connection -> RELU
template <typename DataType>
class SELayer : public BaseLayer<DataType> {
 public:
  SELayer(BaseLayer<DataType>* ip, int numFc1Out,
          bool addPrevLayerBias = false);
  ~SELayer();

  void LoadWeights(float* w1, float* b1, float* w2, float* b2,
                   float* prevLayerBias, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas) override;

 private:
  DataType* w1_ = nullptr;
  DataType* b1_ = nullptr;
  DataType* w2_ = nullptr;
  DataType* b2_ = nullptr;
  DataType* bPrev_ = nullptr;
  int numFc1Out_;
  bool addPrevLayerBias_;
};

template <typename DataType>
class GlobalAvgPoolLayer : public BaseLayer<DataType> {
 public:
  // does averaging of all inputs across W and H dimensions
  // basically get 1 value from the entire 8x8 board plane (squeeze step for SE)
  // output is 2 dimensional containing NxC elements
  GlobalAvgPoolLayer(BaseLayer<DataType>* ip);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas) override;
};

template <typename DataType>
class GlobalScaleLayer : public BaseLayer<DataType> {
 public:
  // scales output (NCHW) with per-channel scaling factors in input2 (NC)
  // also adds input (NCHW)

  GlobalScaleLayer(
      BaseLayer<DataType>* ip);  // ip is pointer to 'input' (of dimension NCHW)

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas) override;
};

// Need memory for 3 data buffers
//  1. input for the layer
//  2. output of the layer
//  3. data from old layer for skip connection

int DivUp(int a, int b) { return (a + b - 1) / b; }

/////////////////////////////////////////////////////////////////////////////
//          Simple CUDA kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void addVectors_kernel(T* c, T* a, T* b, int size, int asize,
                                  int bsize, bool relu, bool useTanh,
                                  bool useSigmoid) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);

    float cVal = aVal + bVal;

    if (relu && (cVal < 0)) cVal = 0;

    if (useTanh) {
      cVal = tanh(cVal);
    }

    if (useSigmoid) {
      cVal = 1.0f / (1.0f + exp(-cVal));
    }

    c[i] = (T)cVal;
  }
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize, bool relu,
                bool use_tanh, bool use_sigmoid) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  addVectors_kernel<<<blocks, kBlockSize>>>(c, a, b, size, asize, bsize, relu,
                                            use_tanh, use_sigmoid);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void addBias_NCHW_kernel(T* c, T* a, T* b, int N, int C, int H,
                                    int W) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int size = N * C * H * W;
  if (i < size) {
    float aVal = (float)a[i];

    // All this math can be optimized, but the kernel is memory bound anyway
    int biasIndex = (i / (H * W)) % C;
    float bVal = (float)b[biasIndex];

    float cVal = aVal + bVal;
    c[i] = (T)cVal;
  }
}

// add bias to convolution's output
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  addBias_NCHW_kernel<<<blocks, kBlockSize>>>(c, a, b, N, C, H, W);
  ReportCUDAErrors(cudaGetLastError());
}


__device__ half readNCHW(float* input_tensor, int n, int c, int h, int w,
                         int Nin, int Cin, int H, int W) {
  if (n >= Nin || c >= Cin) return 0;

  int index;
  index = n;
  index *= Cin;
  index += c;
  index *= H;
  index += h;
  index *= W;
  index += w;

  return (half)(input_tensor[index]);
}

__global__ void fp32NCHWtofp16NHWC_kernel(half* output_tensor,
                                          float* input_tensor, int Nin, int Cin,
                                          int Nout, int Cout, int H, int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= Nout * Cout * H * W) return;

  int index = tid;

  int c = (index % Cout);
  index /= Cout;
  int w = index % W;
  index /= W;
  int h = index % H;
  index /= H;
  int n = index;

  output_tensor[tid] = readNCHW(input_tensor, n, c, h, w, Nin, Cin, H, W);
}

void fp32NCHWtofp16NHWC(half* output_tensor, float* input_tensor, int Nin,
                        int Cin, int Nout, int Cout, int H, int W) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = DivUp(numElements, blockSize);
  fp32NCHWtofp16NHWC_kernel<<<blocks, blockSize>>>(output_tensor, input_tensor,
                                                   Nin, Cin, Nout, Cout, H, W);
}

template <typename DstType, typename SrcType>
__global__ void copyTypeConverted_kernel(DstType* op, SrcType* ip, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N) return;

  DstType el = (DstType)ip[tid];
  op[tid] = el;
}

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  copyTypeConverted_kernel<<<blocks, kBlockSize>>>(op, ip, N);
}

template <typename T>
__global__ void batchNormForward_kernel(T* output, const T* input,
                                        const T* skipInput, int N, int C, int H,
                                        int W, const float* means,
                                        const float* varMultipliers,
                                        bool relu) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  int wIndex = 0;
  if (sizeof(T) == sizeof(float))
    wIndex = (index / (H * W)) % C;  // NCHW for fp32
  else
    wIndex = index % C;  // NHWC for fp16

  float el = input[index];
  float mean = means[wIndex];
  float varMulti = varMultipliers[wIndex];

  el -= mean;
  el *= varMulti;

  if (skipInput) el += (float)skipInput[index];

  if (relu && (el < 0)) el = 0;

  output[index] = (T)el;
}

// Every thread processes single element.
template <typename T>
void batchNormForward(T* output, const T* input, const T* skipInput, int N,
                      int C, int H, int W, float* means, float* var_multipliers,
                      bool relu) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  batchNormForward_kernel<<<blocks, kBlockSize>>>(
      output, input, skipInput, N, C, H, W, means, var_multipliers, relu);

  ReportCUDAErrors(cudaGetLastError());
}

__global__ void expandPlanes_kernel_Fp32_NCHW(float* output,
                                              const uint64_t* masks,
                                              const float* values, int n) {
  // Block size of 256, same mask/val for 64 consecutive threads.
  constexpr int kNumShmemElments = 256 / 64;

  __shared__ uint64_t shMasks[kNumShmemElments];
  __shared__ float shVals[kNumShmemElments];

  int index = threadIdx.x + blockDim.x * blockIdx.x;

  int planeIndex = index >> 6;

  if (planeIndex >= n) return;

  // load inputs to shared memory.
  if (threadIdx.x < kNumShmemElments) {
    shMasks[threadIdx.x] = masks[planeIndex + threadIdx.x];
    shVals[threadIdx.x] = values[planeIndex + threadIdx.x];
  }
  __syncthreads();

  uint64_t mask = shMasks[threadIdx.x >> 6];

  int sqIndex = index & 0x3F;
  float op = 0;

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = shVals[threadIdx.x >> 6];
  }
  output[index] = op;
}

void expandPlanes_Fp32_NCHW(float* output, const uint64_t* masks,
                            const float* values, int n) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int blockSize = 256;
  int blocks = DivUp(threads, blockSize);
  expandPlanes_kernel_Fp32_NCHW<<<blocks, blockSize>>>(output, masks, values,
                                                       n);
  ReportCUDAErrors(cudaGetLastError());
}

// TODO: Can optimize using shared memory if this becomes a bottleneck.
__global__ void expandPlanes_kernel_Fp16_NHWC(half* output,
                                              const uint64_t* masks,
                                              const float* values, int n) {
  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= n * 8 * 8) return;

  const int planeIndex = index % kInputPlanes;
  const int boardIndex = index / (kInputPlanes * 8 * 8);
  const int sqIndex = (index / kInputPlanes) & 0x3F;

  uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

  half op = 0;
  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    float val = values[boardIndex * kInputPlanes + planeIndex];
    op = (half)val;
  }
  output[index] = op;
}

void expandPlanes_Fp16_NHWC(half* output, const uint64_t* masks,
                            const float* values, int n) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  expandPlanes_kernel_Fp16_NHWC<<<blocks, kBlockSize>>>(output, masks, values,
                                                        n);
  ReportCUDAErrors(cudaGetLastError());
}

__global__ void globalScale_kernel(float* output, const float* input,
                                   const float* scale, const float *bias, int inputSize) {
  const int kPlaneSize = 64;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > inputSize) return;

  float val1 = input[tid];   // skip connection
  float val2 = output[tid];  // output of residual block to be scaled

  int sIndex = tid / kPlaneSize;

  float s = scale[sIndex];
  s = 1.0f / (1.0f + exp(-s));  // sigmoid on scale

  float b = bias[sIndex];

  float op = val1 * s + val2 + b;
  if (op < 0) op = 0;
  output[tid] = op;
}

__global__ void globalScale_kernel_fp16_nhwc(half* output, const half* input,
                                             const half* scale, const half* bias, 
                                             int inputSize, int C, int HWC) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > inputSize) return;

  float val1 = (float)input[tid];   // skip connection
  float val2 = (float)output[tid];  // output of residual block to be scaled

  int c = tid % C;
  int n = tid / (HWC);
  int sIndex = n * C + c;
  float s = scale[sIndex];
  s = 1.0f / (1.0f + exp(-s));  // sigmoid on scale

  float b = bias[sIndex];

  float op = val1 * s + val2 + b;
  if (op < 0) op = 0;

  output[tid] = (half)op;
}

// N blocks
// C threads per block
// 'HWC' input data processed by thread block
// each thread writes a single output
__global__ void globalAvgPool_kernel_NHWC_fp16(half* output, const half* input,
                                               int inputSize, int outputSize) {
  const int elementsPerThread = 64;  // 8x8 board

  int blockStart = blockIdx.x * blockDim.x;

  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * blockDim.x + threadIdx.x;
    int inputIndex = blockStart * elementsPerThread + localIndex;
    if (inputIndex < inputSize) S += (float)(input[inputIndex]);
  }

  half avg = (half)(S / elementsPerThread);

  int opIndex = blockStart + threadIdx.x;
  if (opIndex < outputSize) output[opIndex] = avg;
}

// each thread reads 2 inputs (8x8/32), and each warp writes a single output
__global__ void globalAvgPool_kernel(float* output, const float* input,
                                     int inputSize, int outputSize) {
  const int elementsPerWarp = 64;
  const int elementsPerThread = 2;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int laneId = threadIdx.x & 0x1F;
  int laneStartIndex = (tid - laneId) * elementsPerThread;

  // compute per-thread sum for elementsPerThread elements
  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerWarp; i += 32) {
    int index = laneStartIndex + laneId + i;
    if (index < inputSize) S += input[index];
  }

    // compute warp wide sum (for entire plane - elementsPerWarp elements)
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    S += __shfl_down_sync(0xFFFFFFFF, S, offset);
  }

  float avg = S / elementsPerWarp;
  int opIndex = tid >> 5;

  // first thread in warp has the sum, write it in output
  if (laneId == 0) {
    if (opIndex < outputSize) output[opIndex] = avg;
  }
}

#ifdef FUSED_SE_LAYER
// N blocks
// C threads per block
// 'HWC' input data processed by thread block
// each thread processes 8x8 elements
// K is the no. of outputs of first fully connected layer (same as no. of inputs
// for second fully connected layer) the kernel assumes K <= C

// the weights matrix are transposed in reality (K rows, and C columns for fc1,
// and C rows and K columns for fc2)
#define shw1(row, col) (((half*)sharedWeights1)[(col)*C + (row)])
#define shw2(row, col) (((half*)sharedWeights2)[(col)*K + (row)])

template <int C, int K>
__global__ void SE_Layer_NHWC(half* output, const half* skip, const half* input,
                              const half* w1, const half* b1, const half* w2,
                              const half* b2) {
  const int elementsPerThread = 64;  // 8x8 board

  int n = blockIdx.x;
  int c = threadIdx.x;

  __shared__ half sharedData[C];

  half localInput[elementsPerThread];
  half localskip[elementsPerThread];

  // This acutally doesn't save on any global memory reads (each thread block
  // still reads entire weights array redundantly :-/)
  // TODO: can try processing multiple C (multiple planes) in single thread
  // block to get some savings
  //
  // it's only to make the reads faster (better memory coleasing)
  static_assert(((C * K) % 8) == 0, "K*C must be multiple of 8");

  // don't really NEED two shared memory arrays, as the same shared memory can
  // be re-used to hold weights for FC2 after FC1 is done, but loading all
  // weights early seems to improve performance by about 5%
  __shared__ uint4 sharedWeights1[C * K / 8];
  __shared__ uint4 sharedWeights2[C * K / 8];

  half S = 0;

  // 1. global avg (1 avg per thread)
  #pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    localInput[i] = input[inputIndex];
    localskip[i] = skip[inputIndex];
    S += localInput[i];
  }

  half avg = S / (half)elementsPerThread;
  sharedData[c] = avg;

  // load weights for the FC layers in shared memory
  // use uint4 loads to make it faster
  const int numSharedReadsPerThread = K / 8;  // K * C weights, divided by C
                                              // threads, divided by 8 halfs
                                              // (uint4) read per thread
  uint4* w1raw = (uint4*)w1;
  uint4* w2raw = (uint4*)w2;

  #pragma unroll
  for (int i = 0; i < numSharedReadsPerThread; i++) {
    sharedWeights1[c + i * C] = w1raw[c + i * C];
    sharedWeights2[c + i * C] = w2raw[c + i * C];
  }
  __syncthreads();

  // 2. first fully connected layer
  if (c < K) {
    S = 0;

    #pragma unroll
    for (int i = 0; i < C; i++) {
      S += sharedData[i] * shw1(i, c);
    }

    S += b1[c];

    // relu
    if (S < (half)0) S = 0;

    sharedData[c] = S;
  }

  __syncthreads();

  // 3. second fully connected layer
  S = 0;
  #pragma unroll
  for (int i = 0; i < K; i++) {
    S += sharedData[i] * shw2(i, c);
  }
  S += b2[c];

  // sigmoid
  S = (half)(1.0f / (1.0f + exp(-S)));

  // 4. scale, and add skip connection, perform relu, and write to output
  #pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * C + c;
    int inputIndex = n * C * elementsPerThread + localIndex;
    half val = localskip[i] + localInput[i] * S;

    // relu
    if (val < (half)0) val = 0;

    output[inputIndex] = val;
  }
}
#endif
template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : C(c), H(h), W(w), input_(ip) {}

template <typename DataType>
SoftMaxLayer<DataType>::SoftMaxLayer(BaseLayer<DataType>* ip)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip) {
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
}

template <typename DataType>
void SoftMaxLayer<DataType>::Eval(int N, DataType* output,
                                  const DataType* input, const DataType* input2,
                                  void* scratch, size_t scratch_size,
                                  cudnnHandle_t cudnn, cublasHandle_t cublas) {
  float alpha = 1.0f, beta = 0.0f;

  // Need to call this at Eval as 'N' changes :-/
  if (std::is_same<half, DataType>::value) {
    cudnnSetTensor4dDescriptor(out_tensor_desc_, CUDNN_TENSOR_NHWC,
                               CUDNN_DATA_HALF, N, GetC(), GetH(), GetW());
  } else {
    cudnnSetTensor4dDescriptor(out_tensor_desc_, CUDNN_TENSOR_NCHW,
                               CUDNN_DATA_FLOAT, N, GetC(), GetH(), GetW());
  }

  cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, out_tensor_desc_,
                      input, &beta, out_tensor_desc_, output);
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                               int filter, int Cin, bool relu, bool bias)
    : BaseLayer<DataType>(C, H, W, ip),
      filter_size_(filter),
      c_input_(Cin),
      use_relu_(relu),
      use_bias_(bias) {
  // Allocate memory for weights (filter tensor) and biases.
  size_t weight_size = sizeof(DataType) * Cin * C * filter_size_ * filter_size_;
  ReportCUDAErrors(cudaMalloc(&weights, weight_size));

  size_t blas_size = sizeof(DataType) * C;
  ReportCUDAErrors(cudaMalloc(&biases, blas_size));

  const bool fp16 = std::is_same<half, DataType>::value;

  // Create cudnn objects for various tensors, algorithms, etc.
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnCreateConvolutionDescriptor(&conv_desc_);
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
  cudnnCreateTensorDescriptor(&in_tensor_desc_);
  cudnnCreateTensorDescriptor(&bias_desc_);
  cudnnCreateActivationDescriptor(&activation_);

  cudnnSetFilter4dDescriptor(filter_desc_,
                             fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
                             fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
                             GetC(), Cin, filter_size_, filter_size_);

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(
      bias_desc_, fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, 1, C, 1, 1));

  int padding = filter_size_ / 2;
  const bool crossCorr = 1;

  ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
      conv_desc_, padding, padding, 1, 1, 1, 1,
      crossCorr ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION,
      fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT));

  if (fp16)
    ReportCUDNNErrors(
        cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

  // TODO: dynamic selection of algorithm!
  if ((C > 32) && (!fp16)) {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  } else {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }

  if (use_relu_) {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_RELU,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  } else {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_IDENTITY,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
}

template <>
void ConvLayer<half>::LoadWeights(float* pfilter, float* pBias, void* scratch) {
  size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  size_t blas_size = sizeof(float) * C;
  // Also need to convert from fp32 NCHW to fp16 NHWC
  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type / layout conversion using a kernel.
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpyAsync(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  fp32NCHWtofp16NHWC((half*)weights, (float*)scratch, C, c_input_, C, c_input_,
                     filter_size_, filter_size_);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpyAsync(scratch, pBias, blas_size, cudaMemcpyHostToDevice));

    copyTypeConverted((half*)biases, (float*)scratch, C);
  }
}

template <>
void ConvLayer<float>::LoadWeights(float* pfilter, float* pBias,
                                   void* scratch) {
  size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  size_t blas_size = sizeof(float) * C;
  ReportCUDAErrors(
      cudaMemcpyAsync(weights, pfilter, weight_size, cudaMemcpyHostToDevice));

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpyAsync(biases, pBias, blas_size, cudaMemcpyHostToDevice));
  } else {
    ReportCUDAErrors(cudaMemset(biases, blas_size, 0));
  }
}

template <typename DataType>
void ConvLayer<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, cudnnHandle_t cudnn,
                               cublasHandle_t cublas) {
  const bool fp16 = std::is_same<half, DataType>::value;

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(
      out_tensor_desc_, fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, N, C, H, W));

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(
      in_tensor_desc_, fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
      fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, N, c_input_, H, W));

  float alpha = 1.0f, beta = 0.0f;

  if (!(use_relu_ || use_bias_)) {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &beta, out_tensor_desc_,
        output));
  } else if (input2) {
    // fused bias + sum + relu!
    ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &alpha, out_tensor_desc_,
        input2, bias_desc_, biases, activation_, out_tensor_desc_, output));
  } else {
    // For some reason cudnn doesn't support just Convolution + Bias with fp32
    // (winograd algorithm) it works fine when RELU is also needed which is
    // somewhat strange.
    if ((std::is_same<float, DataType>::value) && (!use_relu_)) {
      ReportCUDNNErrors(cudnnConvolutionForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output));
      // add bias
      addBias_NCHW(output, output, biases, N, C, H, W);
    } else {
      ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output, bias_desc_, biases, activation_,
          out_tensor_desc_, output));
    }
  }
}

template <typename DataType>
ConvLayer<DataType>::~ConvLayer() {
  ReportCUDAErrors(cudaFree(weights));
  ReportCUDAErrors(cudaFree(biases));
}

template <typename DataType>
BNLayer<DataType>::BNLayer(BaseLayer<DataType>* ip, bool relu)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      use_relu_(relu) {
  size_t weight_size = sizeof(float) * C;

  ReportCUDAErrors(cudaMalloc(&means_, weight_size));
  ReportCUDAErrors(cudaMalloc(&variances_, weight_size));
}

template <typename DataType>
void BNLayer<DataType>::LoadWeights(float* cpuMeans, float* cpuVar) {
  size_t weight_size = sizeof(float) * C;
  ReportCUDAErrors(
      cudaMemcpyAsync(means_, cpuMeans, weight_size, cudaMemcpyHostToDevice));
  ReportCUDAErrors(
      cudaMemcpyAsync(variances_, cpuVar, weight_size, cudaMemcpyHostToDevice));
}

template <>
void BNLayer<half>::Eval(int N, half* output, const half* input,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t cudnn, cublasHandle_t cublas) {
  batchNormForward(output, input, input2, N, C, H, W, means_, variances_,
                   use_relu_);
}

template <>
void BNLayer<float>::Eval(int N, float* output, const float* input,
                          const float* input2, void* scratch,
                          size_t scratch_size, cudnnHandle_t cudnn,
                          cublasHandle_t cublas) {
  batchNormForward(output, input, input2, N, C, H, W, means_, variances_,
                   use_relu_);
}

template <typename DataType>
BNLayer<DataType>::~BNLayer() {
  ReportCUDAErrors(cudaFree(means_));
  ReportCUDAErrors(cudaFree(variances_));
}

template <typename DataType>
SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int fc1Outputs,
                           bool addPrevLayerBias)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs),
      addPrevLayerBias_(addPrevLayerBias) {
  ReportCUDAErrors(cudaMalloc(&w1_, C * numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&w2_, C * numFc1Out_ * sizeof(DataType)));

  ReportCUDAErrors(cudaMalloc(&b1_, numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&b2_, C * sizeof(DataType)));

  ReportCUDAErrors(cudaMalloc(&bPrev_, C * sizeof(DataType)));
}

template <typename DataType>
SELayer<DataType>::~SELayer() {
  ReportCUDAErrors(cudaFree(w1_));
  ReportCUDAErrors(cudaFree(w2_));
  ReportCUDAErrors(cudaFree(b1_));
  ReportCUDAErrors(cudaFree(b2_));
  ReportCUDAErrors(cudaFree(bPrev_));
}

template <>
void SELayer<float>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                 float* prevLayerBias, void* scratch) {
  // TODO!
}

template <>
void SELayer<half>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                float* prevLayerBias, void* scratch) {
  size_t num_weights = C * numFc1Out_;
  size_t weight_size = sizeof(float) * num_weights;

  // w1
  ReportCUDAErrors(
      cudaMemcpyAsync(scratch, w1, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w1_, (float*)scratch, num_weights);

  // w2
  ReportCUDAErrors(
      cudaMemcpyAsync(scratch, w2, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w2_, (float*)scratch, num_weights);

  // b1
  ReportCUDAErrors(cudaMemcpyAsync(scratch, b1, numFc1Out_ * sizeof(float),
                                   cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b1_, (float*)scratch, numFc1Out_);

  // b2
  ReportCUDAErrors(
      cudaMemcpyAsync(scratch, b2, C * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b2_, (float*)scratch, C);

  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpyAsync(scratch, prevLayerBias, C * sizeof(float),
                                     cudaMemcpyHostToDevice));
    copyTypeConverted((half*)bPrev_, (float*)scratch, C);
  }
}

template <>
void SELayer<float>::Eval(int N, float* output, const float* input,
                          const float* input2, void* scratch,
                          size_t scratch_size, cudnnHandle_t cudnn,
                          cublasHandle_t cublas) {
  // TODO!
}

template <>
void SELayer<half>::Eval(int N, half* output, const half* input,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t cudnn, cublasHandle_t cublas) {
#ifdef FUSED_SE_LAYER
  // TODO: think of more elegant way to avoid this hardcoding :-/
  if (numFc1Out_ == 32) {
    if (C == 64) {
      SE_Layer_NHWC<64, 32>
          <<<N, C>>>(output, input2, input, w1_, b1_, w2_, b2_);
    } else if (C == 128) {
      SE_Layer_NHWC<128, 32>
          <<<N, C>>>(output, input2, input, w1_, b1_, w2_, b2_);
    } else if (C == 192) {
      SE_Layer_NHWC<192, 32>
          <<<N, C>>>(output, input2, input, w1_, b1_, w2_, b2_);
    } else if (C == 256) {
      SE_Layer_NHWC<256, 32>
          <<<N, C>>>(output, input2, input, w1_, b1_, w2_, b2_);
    } else {
      assert(0);  // TODO: support other channel counts
      throw Exception("channel count unsupported by SE layer");
    }
  } else {
    assert(0);  // TODO: support other sizes
    throw Exception("numOutputs unsupported by SE layer");
  }
  ReportCUDAErrors(cudaGetLastError());
#else
  assert(0);
#endif
}

template <typename DataType>
GlobalAvgPoolLayer<DataType>::GlobalAvgPoolLayer(BaseLayer<DataType>* ip)
    : BaseLayer<DataType>(ip->GetC(), 1, 1, ip) {}

template <>
void GlobalAvgPoolLayer<float>::Eval(int N, float* output, const float* input,
                                     const float* input2, void* scratch,
                                     size_t scratch_size, cudnnHandle_t cudnn,
                                     cublasHandle_t cublas) {
  // for NCHW layout (used with fp32),
  // each warp processes a full plane (64 elements), and writes a single
  // average N*C warps are launched
  const int kPlaneSize = input_->GetH() * input_->GetW();
  assert((kPlaneSize % 32) == 0);
  const int kTotalWarps = N * C;
  const int kWarpsPerBlock = 8;
  const int kBlockSize = kWarpsPerBlock * 32;

  int blocks = DivUp(kTotalWarps, kWarpsPerBlock);

  globalAvgPool_kernel<<<blocks, kBlockSize>>>(output, input,
                                               N * C * kPlaneSize, N * C);

  ReportCUDAErrors(cudaGetLastError());
}

template <>
void GlobalAvgPoolLayer<half>::Eval(int N, half* output, const half* input,
                                    const half* input2, void* scratch,
                                    size_t scratch_size, cudnnHandle_t cudnn,
                                    cublasHandle_t cublas) {
  const int kPlaneSize = input_->GetH() * input_->GetW();
  globalAvgPool_kernel_NHWC_fp16<<<N, C>>>(output, input, N * C * kPlaneSize,
                                           N * C);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename DataType>
GlobalScaleLayer<DataType>::GlobalScaleLayer(BaseLayer<DataType>* ip)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip) {}

template <>
void GlobalScaleLayer<float>::Eval(int N, float* output, const float* input,
                                   const float* input2, void* scratch,
                                   size_t scratch_size, cudnnHandle_t cudnn,
                                   cublasHandle_t cublas) {
  // each thread writes one output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * H * W * C, kBlockSize);

  globalScale_kernel<<<kBlocks, kBlockSize>>>(output, input, input2,
                                              input2 + C,
                                              N * C * H * W);

  ReportCUDAErrors(cudaGetLastError());
}

template <>
void GlobalScaleLayer<half>::Eval(int N, half* output, const half* input,
                                  const half* input2, void* scratch,
                                  size_t scratch_size, cudnnHandle_t cudnn,
                                  cublasHandle_t cublas) {
  // each thread writes one output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * H * W * C, kBlockSize);

  globalScale_kernel_fp16_nhwc<<<kBlocks, kBlockSize>>>(
      output, input, input2, input2 + C, N * C * H * W, C, H * W * C);

  ReportCUDAErrors(cudaGetLastError());
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool relu, bool bias, bool tanh, bool sigmoid)
    : BaseLayer<DataType>(C, H, W, ip),
      use_relu_(relu),
      use_bias_(bias),
      use_tanh_(tanh),
      use_sigmoid_(sigmoid) {
  size_t weight_size =
      sizeof(DataType) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  size_t blas_size = sizeof(DataType) * C * H * W;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
  if (use_bias_) {
    ReportCUDAErrors(cudaMalloc(&biases_, blas_size));
  } else {
    biases_ = nullptr;
  }
}

template <>
void FCLayer<half>::LoadWeights(float* cpuWeight, float* cpuBias,
                                void* scratch) {
  size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  size_t weight_size = sizeof(float) * num_weights;
  size_t num_biases = C * H * W;
  size_t blas_size = sizeof(float) * num_biases;

  // also need to convert from fp32 to fp16
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpyAsync(scratch, cpuWeight, weight_size, cudaMemcpyHostToDevice));

  fp32NCHWtofp16NHWC((half*)weights_, (float*)scratch, num_biases,
                     input_->GetC(), num_biases, input_->GetC(), input_->GetH(),
                     input_->GetW());

  if (cpuBias) {
    ReportCUDAErrors(
        cudaMemcpyAsync(scratch, cpuBias, blas_size, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)biases_, (float*)scratch, num_biases);
  }
}

template <>
void FCLayer<float>::LoadWeights(float* cpuWeight, float* cpuBias,
                                 void* scratch) {
  size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  size_t weight_size = sizeof(float) * num_weights;
  size_t num_biases = C * H * W;
  size_t blas_size = sizeof(float) * num_biases;

  ReportCUDAErrors(cudaMemcpyAsync(weights_, cpuWeight, weight_size,
                                   cudaMemcpyHostToDevice));
  if (use_bias_) {
    ReportCUDAErrors(
        cudaMemcpyAsync(biases_, cpuBias, blas_size, cudaMemcpyHostToDevice));
  }
}

// === begin ===
// taken from:
// https://devtalk.nvidia.com/default/topic/883897/error-when-trying-to-use-half-fp16-/
/*
Copyright (c) 2015, Norbert Juffa
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

half uint16_as_fp16(uint16_t a) {
  half res;
#if defined(__cplusplus)
  memcpy(&res, &a, sizeof(res));
#else  /* __cplusplus */
  volatile union {
    __fp16 f;
    uint16_t i;
  } cvt;
  cvt.i = a;
  res = cvt.f;
#endif /* __cplusplus */
  return res;
}

uint32_t fp32_as_uint32(float a) {
  uint32_t res;
#if defined(__cplusplus)
  memcpy(&res, &a, sizeof(res));
#else  /* __cplusplus */
  volatile union {
    float f;
    uint32_t i;
  } cvt;
  cvt.f = a;
  res = cvt.i;
#endif /* __cplusplus */
  return res;
}

/* host version of device function __float2half_rn() */
half float2half_rn(float a) {
  uint32_t ia = fp32_as_uint32(a);
  uint16_t ir;

  ir = (ia >> 16) & 0x8000;
  if ((ia & 0x7f800000) == 0x7f800000) {
    if ((ia & 0x7fffffff) == 0x7f800000) {
      ir |= 0x7c00; /* infinity */
    } else {
      ir = 0x7fff; /* canonical NaN */
    }
  } else if ((ia & 0x7f800000) >= 0x33000000) {
    int shift = (int)((ia >> 23) & 0xff) - 127;
    if (shift > 15) {
      ir |= 0x7c00; /* infinity */
    } else {
      ia = (ia & 0x007fffff) | 0x00800000; /* extract mantissa */
      if (shift < -14) {                   /* denormal */
        ir |= ia >> (-1 - shift);
        ia = ia << (32 - (-1 - shift));
      } else { /* normal */
        ir |= ia >> (24 - 11);
        ia = ia << (32 - (24 - 11));
        ir = ir + ((14 + shift) << 10);
      }
      /* IEEE-754 round to nearest of even */
      if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1))) {
        ir++;
      }
    }
  }
  return uint16_as_fp16(ir);
}

// === end ===

template <>
void FCLayer<half>::Eval(int N, half* output_tensor, const half* input_tensor,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t cudnn, cublasHandle_t cublas) {
  int num_outputs = C * H * W;
  int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  half alpha = float2half_rn(1.0f), beta = float2half_rn(0.0f);
  ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || use_relu_ || use_tanh_ || use_sigmoid_) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, use_relu_, use_tanh_,
               use_sigmoid_);
  }
}

template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* input2,
                          void* scratch, size_t scratch_size,
                          cudnnHandle_t cudnn, cublasHandle_t cublas) {
  int num_outputs = C * H * W;
  int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || use_relu_ || use_tanh_ || use_sigmoid_) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, use_relu_, use_tanh_,
               use_sigmoid_);
  }
}

template <typename DataType>
FCLayer<DataType>::~FCLayer() {
  ReportCUDAErrors(cudaFree(weights_));
  ReportCUDAErrors(cudaFree(biases_));
}

struct InputsOutputs {
  InputsOutputs(int maxBatchSize) {
    ReportCUDAErrors(cudaHostAlloc(
        &input_masks_mem_, maxBatchSize * kInputPlanes * sizeof(uint64_t),
        cudaHostAllocMapped));
    ReportCUDAErrors(
        cudaHostGetDevicePointer(&input_masks_mem_gpu_, input_masks_mem_, 0));

    ReportCUDAErrors(cudaHostAlloc(&input_val_mem_,
                                   maxBatchSize * kInputPlanes * sizeof(float),
                                   cudaHostAllocMapped));
    ReportCUDAErrors(
        cudaHostGetDevicePointer(&input_val_mem_gpu_, input_val_mem_, 0));

    ReportCUDAErrors(cudaHostAlloc(
        &op_policy_mem_, maxBatchSize * kNumOutputPolicy * sizeof(float),
        cudaHostAllocMapped));
    ReportCUDAErrors(
        cudaHostGetDevicePointer(&op_policy_mem_gpu_, op_policy_mem_, 0));

    ReportCUDAErrors(cudaHostAlloc(&op_value_mem_, maxBatchSize * sizeof(float),
                                   cudaHostAllocMapped));
    ReportCUDAErrors(
        cudaHostGetDevicePointer(&op_value_mem_gpu_, op_value_mem_, 0));
  }
  ~InputsOutputs() {
    ReportCUDAErrors(cudaFreeHost(input_masks_mem_));
    ReportCUDAErrors(cudaFreeHost(input_val_mem_));
    ReportCUDAErrors(cudaFreeHost(op_policy_mem_));
    ReportCUDAErrors(cudaFreeHost(op_value_mem_));
  }
  uint64_t* input_masks_mem_;
  float* input_val_mem_;
  float* op_policy_mem_;
  float* op_value_mem_;

  // GPU pointers for the above allocations
  uint64_t* input_masks_mem_gpu_;
  float* input_val_mem_gpu_;
  float* op_policy_mem_gpu_;
  float* op_value_mem_gpu_;
};

// This namespace should be closed at the very end of file, but otherwise
// there are nvcc warnings. Weird way to silence warnings.
}  // namespace

template <typename DataType>
class CudnnNetwork;

template <typename DataType>
class CudnnNetworkComputation : public NetworkComputation {
 public:
  CudnnNetworkComputation(CudnnNetwork<DataType>* network);
  ~CudnnNetworkComputation();

  void AddInput(InputPlanes&& input) override {
    auto iter_mask =
        &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
    auto iter_val =
        &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

    int i = 0;
    for (const auto& plane : input) {
      iter_mask[i] = plane.mask;
      iter_val[i] = plane.value;
      i++;
    }

    batch_size_++;
  }

  void ComputeBlocking() override;

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    return inputs_outputs_->op_value_mem_[sample];
  }
  float GetPVal(int sample, int move_id) const override {
    return inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id];
  }

 private:
  // Memory holding inputs, outputs.
  std::unique_ptr<InputsOutputs> inputs_outputs_;
  int batch_size_;

  CudnnNetwork<DataType>* network_;
};

template <typename DataType>
class CudnnNetwork : public Network {
 public:
  CudnnNetwork(Weights weights, const OptionsDict& options) {
    gpu_id_ = options.GetOrDefault<int>("gpu", 0);

    max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);

    int total_gpus;
    ReportCUDAErrors(cudaGetDeviceCount(&total_gpus));

    if (gpu_id_ >= total_gpus)
      throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

    // Select GPU to run on (for *the current* thread).
    ReportCUDAErrors(cudaSetDevice(gpu_id_));

    ReportCUDNNErrors(cudnnCreate(&cudnn_));
    ReportCUBLASErrors(cublasCreate(&cublas_));

    if (std::is_same<half, DataType>::value) {
      // Check if the GPU support fp16 (Volta+).
      cudaDeviceProp deviceProp = {};
      cudaGetDeviceProperties(&deviceProp, gpu_id_);
      if (deviceProp.major >= 7) {
        // Enable Tensor cores!
        ReportCUBLASErrors(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));
      } else {
        throw Exception("Your GPU doesn't support FP16");
      }
    }

    const int kNumInputPlanes = kInputPlanes;
    const int kNumFilters = weights.input.biases.size();

    numBlocks_ = weights.residual.size();

    has_se_ = false;

    // 0. Process weights.
    processConvBlock(weights.input, true);
    for (auto i = size_t{0}; i < numBlocks_; i++) {
      if (weights.residual[i].has_se) {
        has_se_ = true;
      }
      processConvBlock(weights.residual[i].conv1, true);
      processConvBlock(weights.residual[i].conv2, true);
    }
    processConvBlock(weights.policy);
    processConvBlock(weights.value);

    // 1. Allocate scratch space (used internally by cudnn to run convolutions,
    //     and also for format/layout conversion for weights).
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnTensorDescriptor_t xDesc;
    cudnnCreateFilterDescriptor(&wDesc);
    cudnnCreateConvolutionDescriptor(&convDesc);
    cudnnCreateTensorDescriptor(&xDesc);
    cudnnConvolutionFwdAlgo_t conv_algo;

    const int maxChannels = std::max(kInputPlanes, kNumFilters);

    const bool fp16 = std::is_same<half, DataType>::value;
    ReportCUDNNErrors(cudnnSetFilter4dDescriptor(
        wDesc, fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT,
        fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW, maxChannels, maxChannels,
        3, 3));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(
        xDesc, fp16 ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT, max_batch_size_, maxChannels,
        8, 8));

    ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
        convDesc, 1, 1, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
        fp16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT));

    if (fp16) {
      ReportCUDNNErrors(
          cudnnSetConvolutionMathType(convDesc, CUDNN_TENSOR_OP_MATH));
      conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    } else {
      conv_algo = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    }

    // Query expected scratch space from cudnn.
    ReportCUDNNErrors(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_, xDesc, wDesc, convDesc, xDesc, conv_algo, &scratch_size_));

    // Have some minumum as we also use this for transforming weights.
    const int maxWeightSize = 128 * 1024 * 1024;
    if (scratch_size_ < maxWeightSize) scratch_size_ = maxWeightSize;

    ReportCUDAErrors(cudaMalloc(&scratch_mem_, scratch_size_));
#ifdef DEBUG_RAW_NPS
    printf("\nallocated %d bytes for scratch memory\n", (int)scratch_size_);
#endif

    // 2. Build the network, and copy the weights to GPU memory.

    // input
    {
      auto inputConv = std::make_unique<ConvLayer<DataType>>(
          nullptr, kNumFilters, 8, 8, 3, kNumInputPlanes, true, true);
      inputConv->LoadWeights(&weights.input.weights[0],
                             &weights.input.biases[0], scratch_mem_);
      network_.emplace_back(std::move(inputConv));
    }

    // residual block
    for (int block = 0; block < weights.residual.size(); block++) {
      auto conv1 = std::make_unique<ConvLayer<DataType>>(
          getLastLayer(), kNumFilters, 8, 8, 3, kNumFilters, true, true);
      conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                         &weights.residual[block].conv1.biases[0],
                         scratch_mem_);
      network_.emplace_back(std::move(conv1));

      bool useRelu = weights.residual[block].has_se ? false : true;
      auto conv2 = std::make_unique<ConvLayer<DataType>>(
          getLastLayer(), kNumFilters, 8, 8, 3, kNumFilters, useRelu, true);
      conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                         &weights.residual[block].conv2.biases[0],
                         scratch_mem_);
      BaseLayer<DataType>* conv2Layer = conv2.get();
      network_.emplace_back(std::move(conv2));

      if (weights.residual[block].has_se) {
#ifdef FUSED_SE_LAYER
        int numFCOut = weights.residual[block].se.b1.size();
        auto se = std::make_unique<SELayer<DataType>>(getLastLayer(), numFCOut,
                                                      false);
        se->LoadWeights(&weights.residual[block].se.w1[0],
                        &weights.residual[block].se.b1[0],
                        &weights.residual[block].se.w2[0],
                        &weights.residual[block].se.b2[0], nullptr,
                        scratch_mem_);
        network_.emplace_back(std::move(se));
#else
        auto globalPool =
            std::make_unique<GlobalAvgPoolLayer<DataType>>(getLastLayer());
        network_.emplace_back(std::move(globalPool));

        int numFCOut = weights.residual[block].se.b1.size();
        auto fc1 = std::make_unique<FCLayer<DataType>>(
            getLastLayer(), numFCOut, 1, 1, true, true, false, false);
        fc1->LoadWeights(&weights.residual[block].se.w1[0],
                         &weights.residual[block].se.b1[0], scratch_mem_);
        network_.emplace_back(std::move(fc1));

        auto fc2 = std::make_unique<FCLayer<DataType>>(
            getLastLayer(), kNumFilters * 2, 1, 1, false, true, false, false);
        fc2->LoadWeights(&weights.residual[block].se.w2[0],
                         &weights.residual[block].se.b2[0], scratch_mem_);
        network_.emplace_back(std::move(fc2));

        auto globalScale =
            std::make_unique<GlobalScaleLayer<DataType>>(conv2Layer);
        network_.emplace_back(std::move(globalScale));
#endif
      }
    }

    resi_last_ = getLastLayer();

    // policy head
    {
      auto convPol = std::make_unique<ConvLayer<DataType>>(
          resi_last_, weights.policy.bn_means.size(), 8, 8, 1, kNumFilters);
      convPol->LoadWeights(&weights.policy.weights[0], nullptr, scratch_mem_);
      network_.emplace_back(std::move(convPol));

      auto BNPol = std::make_unique<BNLayer<DataType>>(getLastLayer(), true);
      BNPol->LoadWeights(&weights.policy.bn_means[0],
                         &weights.policy.bn_stddivs[0]);
      network_.emplace_back(std::move(BNPol));

      auto FCPol = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip_pol_b.size(), 1, 1, false, true);
      FCPol->LoadWeights(&weights.ip_pol_w[0], &weights.ip_pol_b[0],
                         scratch_mem_);
      network_.emplace_back(std::move(FCPol));

      auto softmaxPol =
          std::make_unique<SoftMaxLayer<DataType>>(getLastLayer());
      network_.emplace_back(std::move(softmaxPol));
    }
    policy_out_ = getLastLayer();

    // value head
    {
      auto convVal = std::make_unique<ConvLayer<DataType>>(
          resi_last_, weights.value.bn_means.size(), 8, 8, 1, kNumFilters);
      convVal->LoadWeights(&weights.value.weights[0], nullptr, scratch_mem_);
      network_.emplace_back(std::move(convVal));

      auto BNVal = std::make_unique<BNLayer<DataType>>(getLastLayer(), true);
      BNVal->LoadWeights(&weights.value.bn_means[0],
                         &weights.value.bn_stddivs[0]);
      network_.emplace_back(std::move(BNVal));

      auto FCVal1 = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip1_val_b.size(), 1, 1, true, true);
      FCVal1->LoadWeights(&weights.ip1_val_w[0], &weights.ip1_val_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCVal1));

      auto FCVal2 = std::make_unique<FCLayer<DataType>>(getLastLayer(), 1, 1, 1,
                                                        false, true, true);
      FCVal2->LoadWeights(&weights.ip2_val_w[0], &weights.ip2_val_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCVal2));
    }
    value_out_ = getLastLayer();

    // 3. allocate GPU memory for running the network
    //    - three buffers of max size are enough (one to hold input, second to
    //    hold output and third to hold skip connection's input).
    size_t maxSize = resi_last_->GetOutputSize(max_batch_size_);
    for (auto& mem : tensor_mem_) {
      ReportCUDAErrors(cudaMalloc(&mem, maxSize));
      ReportCUDAErrors(cudaMemset(mem, 0, maxSize));
    }

    // printf("Allocated %d bytes of GPU memory to run the network\n", 3 *
    // maxSize);
  }

  void forwardEval(InputsOutputs* io, int batchSize) {
    std::lock_guard<std::mutex> lock(lock_);

#ifdef DEBUG_RAW_NPS
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    // expand packed planes to full planes
    uint64_t* ipDataMasks = io->input_masks_mem_gpu_;
    float* ipDataValues = io->input_val_mem_gpu_;

    if (std::is_same<half, DataType>::value) {
      expandPlanes_Fp16_NHWC((half*)(tensor_mem_[0]), ipDataMasks, ipDataValues,
                             batchSize * kInputPlanes);
    } else {
      expandPlanes_Fp32_NCHW((float*)(tensor_mem_[0]), ipDataMasks,
                             ipDataValues, batchSize * kInputPlanes);
    }

    float* opPol = io->op_policy_mem_gpu_;
    float* opVal = io->op_value_mem_gpu_;

    int l = 0;
    // input
    network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0], nullptr,
                        scratch_mem_, scratch_size_, cudnn_,
                        cublas_);  // input conv

    // residual block
    for (int block = 0; block < numBlocks_; block++) {
      network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                          scratch_mem_, scratch_size_, cudnn_,
                          cublas_);  // conv1

      // for SE Resnet, skip connection is added after SE
      if (has_se_) {
        network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], nullptr,
                            scratch_mem_, scratch_size_, cudnn_,
                            cublas_);  // conv2
      } else {
        network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0],
                            tensor_mem_[2], scratch_mem_, scratch_size_, cudnn_,
                            cublas_);  // conv2
      }

      if (has_se_) {
      // need to preserve both tensor_mem_[1] (op of residual block) and
      // tensor_mem_[2] (skip connection)
#ifdef FUSED_SE_LAYER
        network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[1],
                            tensor_mem_[2], scratch_mem_, scratch_size_, cudnn_,
                            cublas_);  // SE layer
#else
        network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[1], nullptr,
                            scratch_mem_, scratch_size_, cudnn_,
                            cublas_);  // global avg pooling

        network_[l++]->Eval(batchSize, (DataType*)scratch_mem_, tensor_mem_[0],
                            nullptr, scratch_mem_, scratch_size_, cudnn_,
                            cublas_);  // se.fc1

        network_[l++]->Eval(batchSize, tensor_mem_[0], (DataType*)scratch_mem_,
                            nullptr, scratch_mem_, scratch_size_, cudnn_,
                            cublas_);  // se.fc2

        network_[l++]->Eval(
            batchSize, tensor_mem_[2], tensor_mem_[1], tensor_mem_[0],
            scratch_mem_, scratch_size_, cudnn_,
            cublas_);  // global scale + skip connection add + relu

#endif
      }
    }

    // policy head
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                        scratch_mem_, scratch_size_, cudnn_,
                        cublas_);  // pol conv
    network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], nullptr,
                        scratch_mem_, scratch_size_, cudnn_,
                        cublas_);  // pol BN
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[1], nullptr,
                        scratch_mem_, scratch_size_, cudnn_,
                        cublas_);  // pol FC
    if (std::is_same<half, DataType>::value) {
      // TODO: consider softmax layer that writes directly to fp32
      network_[l++]->Eval(batchSize, tensor_mem_[1], tensor_mem_[0], nullptr,
                          scratch_mem_, scratch_size_, cudnn_,
                          cublas_);  // pol softmax
      copyTypeConverted(opPol, (half*)(tensor_mem_[1]),
                        batchSize * kNumOutputPolicy);  // POLICY
    } else {
      network_[l++]->Eval(batchSize, (DataType*)opPol, tensor_mem_[0], nullptr,
                          scratch_mem_, scratch_size_, cudnn_,
                          cublas_);  // pol softmax  // POLICY
    }

    // value head
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                        scratch_mem_, scratch_size_, cudnn_,
                        cublas_);  // value conv
    network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0], nullptr,
                        scratch_mem_, scratch_size_, cudnn_,
                        cublas_);  // value BN
    network_[l++]->Eval(batchSize, tensor_mem_[0], tensor_mem_[2], nullptr,
                        scratch_mem_, scratch_size_, cudnn_,
                        cublas_);  // value FC1

    if (std::is_same<half, DataType>::value) {
      // TODO: consider fusing the bias-add of FC2 with format conversion
      network_[l++]->Eval(batchSize, tensor_mem_[2], tensor_mem_[0], nullptr,
                          scratch_mem_, scratch_size_, cudnn_,
                          cublas_);  // value FC2
      copyTypeConverted(opVal, (half*)(tensor_mem_[2]), batchSize);  // VALUE
    } else {
      network_[l++]->Eval(batchSize, (DataType*)opVal, tensor_mem_[0], nullptr,
                          scratch_mem_, scratch_size_, cudnn_,
                          cublas_);  // value FC2    // VALUE
    }
    ReportCUDAErrors(cudaDeviceSynchronize());

#ifdef DEBUG_RAW_NPS
    const int reportingCalls = 100;
    static int numCalls = 0;
    static int sumBatchSize = 0;
    static double totalTime = 0;

    sumBatchSize += batchSize;
    numCalls++;

    auto t_end = std::chrono::high_resolution_clock::now();

    double dt = std::chrono::duration<double>(t_end - t_start).count();
    totalTime += dt;
    if (numCalls == reportingCalls) {
      double avgBatchSize = ((double)sumBatchSize) / numCalls;
      double nps = sumBatchSize / totalTime;
      printf(
          "\nAvg batch size: %lf, NN eval time: %lf seconds per %d evals. "
          "NPS: "
          "%g\n",
          avgBatchSize, totalTime, sumBatchSize, nps);
      sumBatchSize = 0;
      totalTime = 0;
      numCalls = 0;
    }
#endif
  }

  ~CudnnNetwork() {
    for (auto mem : tensor_mem_) {
      if (mem) ReportCUDAErrors(cudaFree(mem));
    }
    if (scratch_mem_) ReportCUDAErrors(cudaFree(scratch_mem_));
    cudnnDestroy(cudnn_);
    cublasDestroy(cublas_);
  }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    // set correct gpu id for this computation (as it might have been called
    // from a different thread)
    ReportCUDAErrors(cudaSetDevice(gpu_id_));
    return std::make_unique<CudnnNetworkComputation<DataType>>(this);
  }

  std::unique_ptr<InputsOutputs> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      return std::make_unique<InputsOutputs>(max_batch_size_);
    } else {
      std::unique_ptr<InputsOutputs> resource =
          std::move(free_inputs_outputs_.front());
      free_inputs_outputs_.pop_front();
      return resource;
    }
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }

  // Apparently nvcc doesn't see constructor invocations through make_unique.
  // This function invokes constructor just to please complier and silence
  // warning. Is never called (but compiler thinks that it could).
  void UglyFunctionToSilenceNvccWarning() { InputsOutputs io(0); }

 private:
  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;
  int gpu_id_;
  int max_batch_size_;

  // currently only one NN Eval can happen a time (we can fix this if needed
  // by allocating more memory)
  mutable std::mutex lock_;

  int numBlocks_;
  bool has_se_;
  std::vector<std::unique_ptr<BaseLayer<DataType>>> network_;
  BaseLayer<DataType>* getLastLayer() { return network_.back().get(); }

  BaseLayer<DataType>* resi_last_;
  BaseLayer<DataType>* policy_out_;
  BaseLayer<DataType>* value_out_;

  DataType* tensor_mem_[3];
  void* scratch_mem_;
  size_t scratch_size_;

  mutable std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs>> free_inputs_outputs_;

  void processConvBlock(Weights::ConvBlock& block, bool foldBNLayer = false) {
    const float epsilon = 1e-5f;

    // Compute reciprocal of std-dev from the variances (so that it can be
    // just multiplied).
    std::vector<float>& stddev = block.bn_stddivs;
    for (auto&& w : stddev) {
      w = 1.0f / std::sqrt(w + epsilon);
    }

    // Biases are not calculated and are typically zero but some networks
    // might still have non-zero biases. Move biases to batchnorm means to
    // make the output match without having to separately add the biases.
    for (auto j = size_t{0}; j < block.bn_means.size(); j++) {
      block.bn_means[j] -= block.biases[j];
      block.biases[j] = 0.0f;
    }

    // Get rid of the BN layer by adjusting weights and biases of the
    // convolution idea proposed by Henrik Forst�n and first implemented in
    // leela go zero.
    if (foldBNLayer) {
      const int outputs = block.biases.size();
      const int channels = block.weights.size() / (outputs * 3 * 3);

      for (auto o = 0; o < outputs; o++) {
        for (auto c = 0; c < channels; c++) {
          for (auto i = 0; i < 9; i++) {
            block.weights[o * channels * 9 + c * 9 + i] *= block.bn_stddivs[o];
          }
        }

        block.bn_means[o] *= block.bn_stddivs[o];
        block.bn_stddivs[o] = 1.0f;

        // Move means to convolution biases.
        block.biases[o] = -block.bn_means[o];
        block.bn_means[o] = 0.0f;
      }
    }
  }
};

template <typename DataType>
CudnnNetworkComputation<DataType>::CudnnNetworkComputation(
    CudnnNetwork<DataType>* network)
    : network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

template <typename DataType>
CudnnNetworkComputation<DataType>::~CudnnNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

template <typename DataType>
void CudnnNetworkComputation<DataType>::ComputeBlocking() {
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
}

REGISTER_NETWORK("cudnn", CudnnNetwork<float>, 110)
REGISTER_NETWORK("cudnn-fp16", CudnnNetwork<half>, 105)

}  // namespace lczero
