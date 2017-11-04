#ifndef CAFFE_IMAGE_DATA_IM2COL_LAYER_HPP_
#define CAFFE_IMAGE_DATA_IM2COL_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/im2col.hpp"

#include <unistd.h>

#define PARAM_LBN 2048
#define TRIGGER_LBN 4096
#define WRITE_LBN 8192
#define READ_LBN 2097152
#define BYTES_PER_SECTOR 512

namespace caffe {

struct _im2col_param {
	int conv_in_channels;
	int conv_input_shape[2];
	int kernel_shape[2];
	int pad[2];
	int stride[2];
	int dilation[2];
};

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class ImageDataIm2ColLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDataIm2ColLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDataIm2ColLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageDataIm2Col"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 2; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);

  vector<std::pair<std::string, int> > lines_;
  int lines_id_;

// @halfways : im2col
  Blob<int> kernel_shape_;
  Blob<int> stride_;
  Blob<int> pad_;
  Blob<int> dilation_;
  Blob<int> conv_input_shape_;

  vector<int> col_buffer_shape_;
  vector<int> output_shape_;
  vector<int>* bottom_shape_;

  int channel_axis_;
  int num_spatial_axes_;
  int channels_;
  int group_;
  int conv_in_channels_;
  int conv_out_channels_;
  int kernel_dim_;

  int offset_;


  bool force_nd_im2col_;
  bool is_1x1_;


  Blob<Dtype> input_data_;

 private:
  inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
			im2col_cpu(data, conv_in_channels_,
					conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
					kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
					pad_.cpu_data()[0], pad_.cpu_data()[1],
					stride_.cpu_data()[0], stride_.cpu_data()[1],
					dilation_.cpu_data()[0], dilation_.cpu_data()[1], col_buff);
		} else {
    // @halfways : not used for test case, need to find out what nd means
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
					col_buffer_shape_.data(), kernel_shape_.cpu_data(),
					pad_.cpu_data(), stride_.cpu_data(), dilation_.cpu_data(),
					col_buff);
		}
	}

	struct _im2col_param im2col_param;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_DATA_LAYER_IM2COL_HPP_
