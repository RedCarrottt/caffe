#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_im2col_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

// @halfways
#include <fcntl.h>


using namespace std;

namespace caffe {

template <typename Dtype>
ImageDataIm2ColLayer<Dtype>::~ImageDataIm2ColLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void ImageDataIm2ColLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// @halfways : need to check
  const int new_height = this->layer_param_.image_data_im2col_param().new_height();
  const int new_width  = this->layer_param_.image_data_im2col_param().new_width();
  const bool is_color  = this->layer_param_.image_data_im2col_param().is_color();
  string root_folder = this->layer_param_.image_data_im2col_param().root_folder();
  
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_im2col_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  int label;
  while (std::getline(infile, line)) {
    pos = line.find_last_of(' ');
    label = atoi(line.substr(pos + 1).c_str());
    lines_.push_back(std::make_pair(line.substr(0, pos), label));
  }

  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_im2col_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.image_data_im2col_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_im2col_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_im2col_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  // @halfways : InferBlobShape - return shape[4] with cv.img, using param_
  // shape[0] = 1 (N = 1)
  this->transformed_data_.Reshape(top_shape);
  this->input_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_im2col_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  // @halfways : N = batch_size (blob is based on batch)
  // moved to the end of layer setup
  // check point!
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
 

  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
  
// @halfways : im2col setup

  // Configure the kernel size, padding, stride, and inputs.
  ImageDataIm2ColParameter image_im2col_param = this->layer_param_.image_data_im2col_param();
  force_nd_im2col_ = image_im2col_param.force_nd_im2col();
  channel_axis_ = top[0]->CanonicalAxisIndex(image_im2col_param.axis());
  // @halfways : image_im2col_param.axis()
  // With (N, C, H, W) inputs, and axis == 1 (the default), we perform
  // N independent 2D convolutions, sliding C-channel (or (C/g)-channels, for
  // groups g>1) filters across the spatial axes (H, W) of the input.

  const int first_spatial_axis = channel_axis_ + 1;
  const int num_axes = top[0]->num_axes();
  // @halfways : num_axes() - return shape_.size(), default 4 for (N*C*H*W)
  num_spatial_axes_ = num_axes - first_spatial_axis; // 2
  CHECK_GE(num_spatial_axes_, 0);
  vector<int> spatial_dim_blob_shape(1, std::max(num_spatial_axes_, 1)); // 2D
  // Setup filter kernel dimensions (kernel_shape_).
  kernel_shape_.Reshape(spatial_dim_blob_shape);
  int* kernel_shape_data = kernel_shape_.mutable_cpu_data();

  if (image_im2col_param.has_kernel_h() || image_im2col_param.has_kernel_w()) {
    //CHECK_EQ(num_spatial_axes_, 2)
    //    << "kernel_h & kernel_w can only be used for 2D convolution.";
    //CHECK_EQ(0, image_im2col_param.kernel_size_size())
    //    << "Either kernel_size or kernel_h/w should be specified; not both.";
    kernel_shape_data[0] = image_im2col_param.kernel_h();
    kernel_shape_data[1] = image_im2col_param.kernel_w();
  } else {
    const int num_kernel_dims = image_im2col_param.kernel_size_size();
    //CHECK(num_kernel_dims == 1 || num_kernel_dims == num_spatial_axes_)
    //    << "kernel_size must be specified once, or once per spatial dimension "
    //    << "(kernel_size specified " << num_kernel_dims << " times; "
    //    << num_spatial_axes_ << " spatial dims).";
    for (int i = 0; i < num_spatial_axes_; ++i) {
      kernel_shape_data[i] =
	 	  image_im2col_param.kernel_size((num_kernel_dims == 1) ? 0 : i);
    }
  }
  //for (int i = 0; i < num_spatial_axes_; ++i) {
  //  CHECK_GT(kernel_shape_data[i], 0) << "Filter dimensions must be nonzero.";
  //}
  // Setup stride dimensions (stride_).
  stride_.Reshape(spatial_dim_blob_shape);
  int* stride_data = stride_.mutable_cpu_data();
  if (image_im2col_param.has_stride_h() || image_im2col_param.has_stride_w()) {
    //CHECK_EQ(num_spatial_axes_, 2)
    //    << "stride_h & stride_w can only be used for 2D convolution.";
    //CHECK_EQ(0, image_im2col_param.stride_size())
    //    << "Either stride or stride_h/w should be specified; not both.";
    stride_data[0] = image_im2col_param.stride_h();
    stride_data[1] = image_im2col_param.stride_w();
  } else {
    const int num_stride_dims = image_im2col_param.stride_size();
    //CHECK(num_stride_dims == 0 || num_stride_dims == 1 ||
    //      num_stride_dims == num_spatial_axes_)
    //    << "stride must be specified once, or once per spatial dimension "
    //    << "(stride specified " << num_stride_dims << " times; "
    //    << num_spatial_axes_ << " spatial dims).";
    const int kDefaultStride = 1;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      stride_data[i] = (num_stride_dims == 0) ? kDefaultStride :
          image_im2col_param.stride((num_stride_dims == 1) ? 0 : i);
    //  CHECK_GT(stride_data[i], 0) << "Stride dimensions must be nonzero.";
    }
  }
  // Setup pad dimensions (pad_).
  pad_.Reshape(spatial_dim_blob_shape);
  int* pad_data = pad_.mutable_cpu_data();
  if (image_im2col_param.has_pad_h() || image_im2col_param.has_pad_w()) {
    //CHECK_EQ(num_spatial_axes_, 2)
    //    << "pad_h & pad_w can only be used for 2D convolution.";
    //CHECK_EQ(0, image_im2col_param.pad_size())
    //    << "Either pad or pad_h/w should be specified; not both.";
    pad_data[0] = image_im2col_param.pad_h();
    pad_data[1] = image_im2col_param.pad_w();
  } else {
    const int num_pad_dims = image_im2col_param.pad_size();
    //CHECK(num_pad_dims == 0 || num_pad_dims == 1 ||
    //     num_pad_dims == num_spatial_axes_)
    //    << "pad must be specified once, or once per spatial dimension "
    //    << "(pad specified " << num_pad_dims << " times; "
    //    << num_spatial_axes_ << " spatial dims).";
    const int kDefaultPad = 0;
    for (int i = 0; i < num_spatial_axes_; ++i) {
      pad_data[i] = (num_pad_dims == 0) ? kDefaultPad :
          image_im2col_param.pad((num_pad_dims == 1) ? 0 : i);
    }
  }
  // Setup dilation dimensions (dilation_).
  dilation_.Reshape(spatial_dim_blob_shape);
  int* dilation_data = dilation_.mutable_cpu_data();
  const int num_dilation_dims = image_im2col_param.dilation_size();
  //CHECK(num_dilation_dims == 0 || num_dilation_dims == 1 ||
  //      num_dilation_dims == num_spatial_axes_)
  //    << "dilation must be specified once, or once per spatial dimension "
  //    << "(dilation specified " << num_dilation_dims << " times; "
  //    << num_spatial_axes_ << " spatial dims).";
  const int kDefaultDilation = 1;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    dilation_data[i] = (num_dilation_dims == 0) ? kDefaultDilation :
                       image_im2col_param.dilation((num_dilation_dims == 1) ? 0 : i);
  }
  // Special case: im2col is the identity for 1x1 convolution with stride 1
  // and no padding, so flag for skipping the buffer and transformation.
  is_1x1_ = true;
  for (int i = 0; i < num_spatial_axes_; ++i) {
    is_1x1_ &=
        kernel_shape_data[i] == 1 && stride_data[i] == 1 && pad_data[i] == 0;
    if (!is_1x1_) { break; }
  }

  // Configure output channels and groups.
  channels_ = top[0]->shape(channel_axis_);
  group_ = image_im2col_param.group();
  //CHECK_EQ(channels_ % group_, 0);
  //    << "Number of output should be multiples of group.";
  conv_in_channels_ = channels_;
  //conv_out_channels_ = num_output_;

  // @halfways : kernel(filter) c * h * w
  // cannot figure out why kernel_dim_ must be applied to im2col output size
  kernel_dim_ = conv_in_channels_ 
	  * kernel_shape_data[0] * kernel_shape_data[1];

  // @halfways : from void BaseConvolutionLayer<Dtype>::Reshape
  // ignore any exceptional case

  //num_ = top[0]->count(0, channel_axis_); // not used

  // Setup input dimensions (conv_input_shape_).
  vector<int> top_dim_blob_shape(1, num_spatial_axes_ + 1);
  conv_input_shape_.Reshape(top_dim_blob_shape);
  int* conv_input_shape_data = conv_input_shape_.mutable_cpu_data();
  // num_spatial_axes = 2
  // channel_axis_ = 1
  // conv_input_shape_[1] = top[0]->shape[2] (H)
  // conv_input_shape_[2] = top[0]->shape[3] (W)  
  for (int i = 0; i < num_spatial_axes_ + 1; ++i) {
    conv_input_shape_data[i] = top[0]->shape(channel_axis_ + i);
  }

  // The im2col result buffer will only hold one image at a time to avoid
  // overly large memory usage. In the special case of 1x1 convolution
  // it goes lazily unused to save memory.

  // @halfways : usage of col_buffer_; direct access for data_?

  offset_ = 1;
  col_buffer_shape_.clear();
  col_buffer_shape_.push_back(batch_size);
  // kernel_dim_ = kernel C * H * W
  // why is it needed?
  col_buffer_shape_.push_back(kernel_dim_);
  offset_ *= kernel_dim_;
  // OH = (H + 2P - FH) / S + 1
  // OW = (W + 2P - FO) / S + 1
  for (int i = 0; i < num_spatial_axes_; ++i) {
    const int* kernel_shape_data = kernel_shape_.cpu_data();
	const int* stride_data = stride_.cpu_data();
	const int* pad_data = pad_.cpu_data();
	const int* dilation_data = dilation_.cpu_data();
	
	const int input_dim = top_shape[i + 2];
	const int kernel_extent = dilation_data[i] 
		* (kernel_shape_data[i] - 1) + 1; // d * (k_h - 1) + 1?
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
	   	/ stride_data[i] + 1;
	col_buffer_shape_.push_back(output_dim);
	offset_ *= output_dim;
  }

  // @halfways : first set data_ size for im2col
  // as capacity of data_ never shrink, it will sustain the size
  // if once enlarged
  for (int i = 0; i < this->prefetch_.size(); ++i) {
	  this->prefetch_[i]->data_.Reshape(col_buffer_shape_);
  }
  top[0]->Reshape(col_buffer_shape_);

  // not used
  //bottom_dim_ = bottom[0]->count(channel_axis_); 
  //top_dim_ = top[0]->count(channel_axis_);

  // @halfways : N = batch_size (blob is based on batch)
  // needs to move this part to the end of layer setup
  // check point!
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
	  this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
 	  << top[0]->channels() << "," << top[0]->height() << ","
 	  << top[0]->width();
  
  /*
  // label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
  */
// @halfways : check point - col_buffer_ + kernel_dim_
}

template <typename Dtype>
void ImageDataIm2ColLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataIm2ColLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataIm2ColParameter image_data_im2col_param = this->layer_param_.image_data_im2col_param();
  const int batch_size = image_data_im2col_param.batch_size();
  const int new_height = image_data_im2col_param.new_height();
  const int new_width = image_data_im2col_param.new_width();
  const bool is_color = image_data_im2col_param.is_color();
  string root_folder = image_data_im2col_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  //this->transformed_data_.Reshape(top_shape);
  this->input_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
	//int offset = batch->data_.offset(item_id); // offset = C * H * W
    //this->transformed_data_.set_cpu_data(prefetch_data + offset);
    //this->data_transformer_->Transform(cv_img, &(this->transformed_data_));

	// @halfways : Transform -> im2col with direct access to prefetch_data
	// 1) Transform to get input of im2col in temp
	// 2) put into im2col to get col
	
	int offset = offset_ * item_id;
	this->transformed_data_.set_cpu_data(prefetch_data + offset);
	this->data_transformer_->Transform(cv_img, &(this->input_data_));
	const Dtype* input_data_ptr = this->input_data_.cpu_data();
	Dtype* transformed_data_ptr = 
		this->transformed_data_.mutable_cpu_data();
	
	// fd open
	int fd_cv_img = open("/dev/sdb2", O_RDWR);
	int fd_param = open("/dev/sdb1", O_RDWR);
	int fd_read_im2col = open("/dev/sdb3", O_RDWR);
	lseek(fd_cv_img, 0, SEEK_SET);	
	lseek(fd_param, 0, SEEK_SET);
	lseek(fd_read_im2col, 0, SEEK_SET);

	size_t tmp;

	// first write cv_img for test
	// this should be removed for real use
	tmp = write(fd_cv_img, input_data_ptr, 
			sizeof(Dtype) * input_data_.count());
	fsync(fd_cv_img);

	// read test
	/*
	Dtype* read_test = (Dtype*)malloc(sizeof(Dtype) * input_data_.count());
	lseek(fd_cv_img, 0, SEEK_SET);
	read(fd_cv_img, read_test, sizeof(Dtype) * input_data_.count());
	*/
	
	// make im2col_param
	struct _im2col_param im2col_param;
	im2col_param.conv_in_channels = conv_in_channels_;
	for(int i = 0; i < 2; ++i) {
		im2col_param.conv_input_shape[i] =
		   conv_input_shape_.cpu_data()[i + 1];
		im2col_param.kernel_shape[i] = kernel_shape_.cpu_data()[i];
		im2col_param.pad[i] = pad_.cpu_data()[i];
		im2col_param.stride[i] = stride_.cpu_data()[i];
		im2col_param.dilation[i] = dilation_.cpu_data()[i];
	}
	
	// write param
	tmp = write(fd_param, &im2col_param, sizeof(im2col_param));
	fsync(fd_param);
	// write param test
	/*
	lseek(fd_param, 0, SEEK_SET); 
	struct _im2col_param* test =
		(struct _im2col_param*)malloc(sizeof(struct _im2col_param));
	tmp = read(fd_param, test, sizeof(struct _im2col_param));
	cout << "in im2col_param : " << im2col_param.dilation[0] << endl;
	cout << "test : " << test->dilation[0] << endl;
	*/

	// read im2col
	//tmp = read(fd_read_im2col, transformed_data_ptr,
	//		this->transformed_data_.count());

	// fd close
	close(fd_cv_img);
	close(fd_param);
	close(fd_read_im2col);
	
	conv_im2col_cpu(input_data_ptr, transformed_data_ptr);
	
	// @halfways : test cout for image file
	/*
	int top_index;
	for(int h = 0; h < cv_img.rows; ++h) {
		for(int w = 0; w < cv_img.cols; ++h) {
			for(int c = 0; c < cv_img.channels(); ++c) {
				top_index = (c * cv_img.rows + h) * cv_img.cols + w;
				cout << input_data_ptr[top_index] << " : ";
				cout << read_test[top_index] << endl;
			}
			break;
		}
		break;
	}
	*/
	
	
	//cout << *transformed_data_ptr << endl;

	trans_time += timer.MicroSeconds();

    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_im2col_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataIm2ColLayer);
REGISTER_LAYER_CLASS(ImageDataIm2Col);

}  // namespace caffe
#endif  // USE_OPENCV
