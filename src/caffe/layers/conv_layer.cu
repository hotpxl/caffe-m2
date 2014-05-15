// Copyright 2013 Yangqing Jia

#include <vector>
#include <cstdio>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

cudaStream_t calcStream[2] = {0, 0};
extern int lock;

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  // First, im2col
  im2col_gpu(bottom_data, NUM_, CHANNELS_, HEIGHT_,
                    WIDTH_, KSIZE_, PAD_, STRIDE_, col_data);
  // Second, innerproduct with groups
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
      (Dtype)1., weight, col_data,
      (Dtype)0., top_data);
  // third, add bias
  if (biasterm_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NUM_OUTPUT_,
        N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        (Dtype)1., top_data);
  }
}

template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  const Dtype* bottom_data = (*bottom)[0]->gpu_data();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  Dtype* col_data = col_buffer_.mutable_gpu_data();
  Dtype* col_diff = col_buffer_.mutable_gpu_diff();
  // bias gradient if necessary
  Dtype* bias_diff = NULL;

  if (biasterm_) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_gemv<Dtype>(CblasNoTrans, NUM_OUTPUT_, N_,
        1., top_diff,
        reinterpret_cast<const Dtype*>(bias_multiplier_->gpu_data()),
        0., bias_diff);
  }
  // since we saved memory in the forward pass by not storing all col data,
  // we will need to recompute them.
  // im2col_gpu(bottom_data, NUM_, CHANNELS_, HEIGHT_,
  //                  WIDTH_, KSIZE_, PAD_, STRIDE_, col_data);
  // gradient w.r.t. weight. Note that we will accumulate diffs.
  if (!calcStream[0]) {
    CUDA_CHECK(cudaStreamCreate(&calcStream[0]));
    CUDA_CHECK(cudaStreamCreate(&calcStream[1]));
  }
  lock = 1;
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
      (Dtype)1., top_diff,
      col_data, (Dtype)0.,
      weight_diff, calcStream[0]);
  // gradient w.r.t. bottom data, if necessary
  if (propagate_down) {
    caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
      (Dtype)1., weight,
      top_diff,
      (Dtype)0., col_diff, calcStream[1]);
    // col2im back to the data
    col2im_gpu(col_diff, NUM_, CHANNELS_, HEIGHT_, WIDTH_, KSIZE_, PAD_, STRIDE_,
        bottom_diff);
  }
  return Dtype(0.);
}


INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
