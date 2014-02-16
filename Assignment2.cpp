/// Homework 2
/// Image Blurring
//****************************************************************************
/// Reference:
/// http://forums.udacity.com/questions/100028762/an-overview-of-hw2-how-to-think-about-the-problem#cs344
//
/// A good starting place is to map each thread to a pixel as you have before.
/// Then every thread can perform steps 2 and 3 in the diagram above
/// completely independently of one another.

/// Note that the array of weights is square, so its height is the same as its width.
/// We refer to the array of weights as a filter, and we refer to its width with the
/// variable filterWidth.

//****************************************************************************
/// Once you have gotten that working correctly, then you can think about using
/// shared memory and having the threads cooperate to achieve better performance.
//****************************************************************************
/// Here is an example of the safe way to do the same thing:
/// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//
/// Finally, remember to free the memory you allocate at the end of the function.
//****************************************************************************

#include "reference_calc.cpp"
#include "utils.h"

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols,
                   const float* const filter, const int filterWidth)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_pos.y >= numRows || thread_2D_pos.x >= numCols)
    return;
  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  assert(filterWidth % 2 == 1);
  float outputPixel = 0.f;
  for (int filterRow = -filterWidth/2; filterRow <= filterWidth/2; ++filterRow) {
    for (int filterCol = -filterWidth/2; filterCol <= filterWidth/2; ++filterCol) {
      int neighborRow = min(numRows - 1, max(0, thread_2D_pos.y + filterRow));
      int neighborCol = min(numCols - 1, max(0, thread_2D_pos.x + filterCol));
      int neighbor_1D_pos = neighborRow * numCols + neighborCol;
      float neighbor = static_cast<float>(inputChannel[neighbor_1D_pos]);

      int filter_pos = (filterRow + filterWidth/2) * filterWidth + filterCol + filterWidth/2;
      float filter_value = filter[filter_pos];

      outputPixel += neighbor * filter_value;
    }
  }
  outputChannel[thread_1D_pos] = outputPixel;
  // NOTE: If a thread's absolute position 2D position is within the image, but some of
  // its neighbors are outside the image, then you will need to be extra careful. Instead
  // of trying to read such a neighbor value from GPU memory (which won't work because
  // the value is out of bounds), you should explicitly clamp the neighbor values you read
  // to be within the bounds of the image. If this is not clear to you, then please refer
  // to sequential reference solution for the exact clamping semantics you should follow.
}

__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  uchar4 currentPixel = inputImageRGBA[thread_1D_pos];

  redChannel[thread_1D_pos]   = currentPixel.x;
  greenChannel[thread_1D_pos] = currentPixel.y;
  blueChannel[thread_1D_pos]  = currentPixel.z;
}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);
  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
                                const float* const h_filter, const size_t filterWidth)
{
  //allocate memory for the three different channels
  //original
  checkCudaErrors(cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
  checkCudaErrors(cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage));

  //TODO:
  //Allocate memory for the filter on the GPU
  //Use the pointer d_filter that we have already declared for you
  //You need to allocate memory for the filter with cudaMalloc
  //be sure to use checkCudaErrors like the above examples to
  //be able to tell if anything goes wrong
  //IMPORTANT: Notice that we pass a pointer to a pointer to cudaMalloc
  checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth * filterWidth));

  //TODO:
  //Copy the filter on the host (h_filter) to the memory you just allocated
  //on the GPU.  cudaMemcpy(dst, src, numBytes, cudaMemcpyHostToDevice);
  //Remember to use checkCudaErrors!
  checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float) * filterWidth * filterWidth,
                  cudaMemcpyHostToDevice));
}

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
                        uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
                        unsigned char *d_redBlurred,
                        unsigned char *d_greenBlurred,
                        unsigned char *d_blueBlurred,
                        const int filterWidth)
{
  //TODO: Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(32, 32, 1);
  //TODO:
  //Compute correct grid size (i.e., number of blocks per kernel launch)
  //from the image size and and block size.
  const dim3 gridSize(numCols/32 + 1, numRows/32 + 1, 1);
  //TODO: Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA,
                                            numRows,
                                            numCols,
                                            d_red,
                                            d_green,
                                            d_blue);

  // Call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  //TODO: Call your convolution kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred,
                                         numRows, numCols,
                                         d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred,
                                         numRows, numCols,
                                         d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred,
                                         numRows, numCols,
                                         d_filter, filterWidth);
  // Again, call cudaDeviceSynchronize(), then call checkCudaErrors() immediately after
  // launching your kernel to make sure that you didn't make any mistakes.
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  // Now we recombine your results. We take care of launching this kernel for you.
  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

//Free all the memory that we allocated
void cleanup() {
  checkCudaErrors(cudaFree(d_red));
  checkCudaErrors(cudaFree(d_green));
  checkCudaErrors(cudaFree(d_blue));
  checkCudaErrors(cudaFree(d_filter));
}
