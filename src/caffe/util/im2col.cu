// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

namespace caffe {

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im, const int num,
    const int height, const int width, const int ksize, const int channelNum, const int pad,
    const int stride, const int height_col, const int width_col,
    Dtype* data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    int h_out = index / width_col % height_col;
    int imageIdx = index / width_col / height_col % num;
    int channel_in = index / width_col / height_col / num;
    int channel_out = channel_in * ksize * ksize;
    int h_in = h_out * stride - pad;
    int w_in = w_out * stride - pad;
    data_col += ((channel_out * num + imageIdx) * height_col + h_out) * width_col + w_out;
    data_im += ((imageIdx * channelNum + channel_in) * height + h_in) * width + w_in;
    for (int i = 0; i < ksize; ++i) {
      for (int j = 0; j < ksize; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
            data_im[i * width + j] : 0;
        data_col += height_col * width_col * num;
      }
    }
  }
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int num, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = num * channels * height_col * width_col;
  // NOLINT_NEXT_LINE(whitespace/operators)
  im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_im, num, height, width, ksize, channels, pad, stride, height_col,
      width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int num, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, float* data_col);
template void im2col_gpu<double>(const double* data_im, const int num, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col, const int num,
    const int height, const int width, const int channels, const int ksize, const int channelNum,
    const int pad, const int stride, const int height_col, const int width_col,
    Dtype* data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype val = 0;
    int w = index % width + pad;
    int h = (index / width) % height + pad;
    int c = index / (width * height) % channelNum;
    int imageIdx = index / channelNum / width / height;
    // compute the start and end of the output
    int w_col_start = (w < ksize) ? 0 : (w - ksize) / stride + 1;
    int w_col_end = min(w / stride + 1, width_col);
    int h_col_start = (h < ksize) ? 0 : (h - ksize) / stride + 1;
    int h_col_end = min(h / stride + 1, height_col);
    /*
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        // the col location: [c * width * height + h_out, w_out]
        int c_col = c * ksize * ksize + (h - h_col * stride) * ksize + (w - w_col * stride);
        val += data_col[(c_col * height_col + h_col) * width_col + w_col];
      }
    }
    */
    // equivalent implementation
    int offset = ((c * ksize * ksize + h * ksize + w) * num + imageIdx) * height_col * width_col;
    int coeff_h_col = width_col - stride * ksize * height_col * width_col * num;
    int coeff_w_col = 1 - stride * height_col * width_col * num;
    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
      for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
        val += data_col[offset + h_col * coeff_h_col + w_col * coeff_w_col];
      }
    }
    data_im[index] = val;
  }
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int num, const int channels,
    const int height, const int width, const int ksize, const int pad,
    const int stride, Dtype* data_im) {
  // CUDA_CHECK(cudaMemset(data_im, 0,
  //            sizeof(Dtype) * height * width * channels));
  int height_col = (height + 2 * pad - ksize) / stride + 1;
  int width_col = (width + 2 * pad - ksize) / stride + 1;
  int num_kernels = num * channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  // NOLINT_NEXT_LINE(whitespace/operators)
  col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
                             CAFFE_CUDA_NUM_THREADS>>>(
      num_kernels, data_col, num, height, width, channels, ksize, channels, pad, stride,
      height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int num, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int num, const int channels,
    const int height, const int width, const int psize, const int pad,
    const int stride, double* data_im);


}  // namespace caffe
