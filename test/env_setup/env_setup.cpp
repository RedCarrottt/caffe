#include <fstream>
#include <iostream>
#include <stdio.h>
#include <fcntl.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <vector>
#include <utility>
#include <string>
#include <unistd.h>

//#include "caffe/common.hpp"
//#include "caffe/util/io.hpp"

using namespace std;


cv::Mat ReadImageToCVMat (const char* filename, 
		const int height, const int width, const bool is_color) {
	cv::Mat cv_img;
	int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
	
	if(!cv_img_origin.data) {
		//LOG(ERROR) << "failed to open file : " << filename;
		return cv_img_origin;
	}
	if(height > 0 && width > 0) {
		cv::resize(cv_img_origin, cv_img, cv::Size(width, height));
	} else {
		cv_img = cv_img_origin;
	}
	return cv_img;
}


int main() {
	char root_folder[1024], source[1024];
    //int new_height = 256, new_width = 256;
	//bool is_color = true;
	
	size_t size = 0;

	getcwd(root_folder, 1024);
	strcpy(source, "cat.jpg");

	int fd_cv_img = open("/dev/sdb2", O_RDWR);
	//void* read_cv_img = (cv::Mat*)malloc(sizeof(cv::Mat));	

	//printf("%s/%s\n", root_folder, source);
	strcat(root_folder, "/");
	strcat(root_folder, source);
	strcpy(source, root_folder);
	//printf("%s\n", source);
	cv::Mat cv_img = ReadImageToCVMat(source, 0, 0, true);
	cv::Mat* read_cv_img = (cv::Mat*)malloc(sizeof(cv::Mat));	
	//printf("%s/%s\n", root_folder, source);

	lseek(fd_cv_img, 0, SEEK_SET);
	cv::Mat* ptr_cv_img = &cv_img;
	//write(fd_cv_img, ptr_cv_img, sizeof(cv_img));
	
	size = sizeof(cv_img.cols) + sizeof(cv_img.data) + sizeof(cv_img.dataend) + sizeof(cv_img.datalimit) + sizeof(cv_img.datastart)
		+ sizeof(cv_img.dims) + sizeof(cv_img.flags) + sizeof(cv_img.rows) + sizeof(cv_img.size) + sizeof(cv_img.step) + sizeof(cv_img.allocator)
		+ sizeof(cv_img.refcount);
	
	/*
	cout << "cols : " << cv_img.cols << endl;
	cout << "rows : " << cv_img.rows << endl;
	cout << "dims : " << cv_img.dims << endl;
	cout << "flags : " << cv_img.flags << endl;
	cout << "size : " << *cv_img.size << endl;
	cout << "step : " << cv_img.step << endl;
	cout << "data : " << sizeof(cv_img.data) << endl;
	//cout << "datastart : " << cv_img.datastart << endl;
	//cout << "dataend : " << cv_img.dataend << endl;
	//cout << "datalimit : " << cv_img.datalimit << endl;
	//cout << "allocator : " << cv_img.allocator << endl;
	cout << "refcount : " << *cv_img.refcount << endl;
    */

	// @halfways : test part
	lseek(fd_cv_img, 0, SEEK_SET);
	read(fd_cv_img, read_cv_img, sizeof(cv_img));
	cout << read_cv_img->channels() << endl;

	//cout << cv_img.channels() << endl;
	//cout << cv_img.rows << endl;
	//cout << cv_img.cols << endl;

	int count = cv_img.channels() * cv_img.rows * cv_img.cols;
		

    // @halfways : blob making
	
    //Dtype* transformed_data = (Dtype*)malloc(count * sizeof(Dtype));
	/*
	int top_index;
	for (int h = 0; h < height; ++h) {
	const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
	int img_index = 0;
	for (int w = 0; w < width; ++w) {
		for (int c = 0; c < img_channels; ++c) {
			if (do_mirror) {
				top_index = (c * height + h) * width + (width - 1 - w);
			} else {
				top_index = (c * height + h) * width + w;
		    }
			// int top_index = (c * height + h) * width + w;
		  	Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
   			if (has_mean_file) {
	   			int mean_index = (c * img_height + h_off + h) * img_width
				   	+ w_off + w;
				transformed_data[top_index] =
					(pixel - mean[mean_index]) * scale;
			} else {
	  			if (has_mean_values) {
					transformed_data[top_index] =
		  				(pixel - mean_values_[c]) * scale;
	  			} else {
					transformed_data[top_index] = pixel * scale;
	 			}
   			}
  		}
	}
*/
	//cout << "size : " << sizeof(transformed_data) << endl;
	//cout << "size : " << sizeof(*transforemd_data) << endl;

	close(fd_cv_img);

	return 0;
}


/*
   Class Mat -> /usr/include/opencv2/core/core.hpp
*/
