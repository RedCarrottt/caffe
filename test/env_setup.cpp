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

namespace test {

cv::Mat ReadImageToCVMat (const char[] filename, 
		const int height, const int width, const bool is_color) {
	cv::Mat cv_img;
	int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR : CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat cv_img_origin = cv::imread(filename, cv_read_flag);
	
	if(!cv_img_origin.data) {
		LOG(ERROR) << "failed to open file : " << filename;
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

	getcwd(root_folder, 1024);
	strcpy(source, "cat.jpg");

	int fd_cv_img = open("/dev/sdb2", O_RDWR);
	
	//printf("%s/%s\n", root_folder, source);
	strcat(root_folder, "/");
	strcat(root_folder, source);
	strcpy(source, root_folder);
	//printf("%s\n", source);
	cv::Mat cv_img = ReadImageToCVMat(source, 0, 0, true);
    


	return 0;
}

}
