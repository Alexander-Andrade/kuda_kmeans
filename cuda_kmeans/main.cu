#include <cuda.h>
#include<cuda_runtime_api.h>
#include "Includes.h"
#include <thrust\device_vector.h>
#include <thrust\host_vector.h>
#include <functional>
#include <algorithm>
#include <utility>

using namespace std;
using namespace thrust;

using himage_t = vector<float>;
using himage_set_t = vector<himage_t>;
using label_t = unsigned char;
using hlabel_set_t = vector<label_t>;

//random engine
std::default_random_engine randGen(std::time(NULL));
std::uniform_real_distribution<float> unifDistr(0.0, 1.0);
auto randFloat = std::bind(unifDistr, randGen);

int determFeatureNum(istream& stream,char delim=','){
	long long cur_pos = stream.tellg();
	stream.seekg(0, stream.beg);
	
	string line;
	std::getline(stream,line);

	stringstream ss(line);
	int n_features = 0;
	
	float num_buf;
	bool end_line;

	while (true){
		if (ss >> num_buf){
			n_features++;
			end_line = false;
		}
		else{
			if (end_line) break;
			ss.clear();
			ss.ignore(10, delim);
			end_line = true;
		}
	}
	stream.seekg(cur_pos, stream.beg);
	return n_features;
}

himage_set_t imagesFromFile(istream& stream, int estimated_vec_num = 1024, char delim = ',', int n_ignored = 100){
	himage_set_t fileImages;
	fileImages.reserve(estimated_vec_num);
	int nImageFeatures = determFeatureNum(stream);
	float curFeature = 0;
	while (!stream.eof()){
		himage_t curImage(nImageFeatures);
		for (int i = 0; i < nImageFeatures;)
			if (stream >> curFeature){
				curImage[i] = curFeature;
				i++;
				i == nImageFeatures ? stream.ignore(n_ignored, '\n') : stream.ignore(n_ignored, delim);
			}
			else{
				//if there is non number column in the table (in the center)
				if (stream.eof()) break;
				stream.clear();
				stream.ignore(n_ignored, delim);
			}
			fileImages.push_back(curImage);
	}
	return fileImages;
}

std::pair<float, float> featureMinMax(himage_set_t& images, int featureNo){
	float min = images[0][featureNo];
	float max = min;
	float curFeature = 0;
	for (auto& image: images){
		curFeature = image[featureNo];
			if(curFeature > max)
				max = curFeature;
			if (curFeature < min)
				min = curFeature;
	}
	return std::make_pair(min, max);
}

void normalizeFeature(himage_set_t& images, int featureNo){
	std::pair<float, float> minmax = featureMinMax(images, featureNo);
	float min = minmax.first;
	float delta = minmax.second - minmax.first;
	for (auto& image : images)
		image[featureNo] = (image[featureNo] - min) / delta;
}

void normalizeImages(himage_set_t& images){
	int nImageFeatures = images[0].size();
	for (int i = 0; i < nImageFeatures; i++){
		normalizeFeature(images, i);
	}
}

void randomizeImage(himage_t& image){
	for (auto& feature : image)
		feature = unifDistr(randGen);
}

double euclideanDistance(himage_t& image1, himage_t image2){
	double squaredDiffSum = 0.0;
	int nImageFeatures = image1.size();
	for (int i = 0; i < nImageFeatures; i++)
		squaredDiffSum += pow(image1[i] - image2[i], 2);
	return sqrt(squaredDiffSum);
}

byte getClasterLabel(himage_t& image, himage_set_t& clastersCenters){
	double minDist = euclideanDistance(image, clastersCenters[0]);
	double curDist = 0.0;
	int clasterLabel = 0;
	int nClasters = clastersCenters.size();
	for (int i = 1; i < nClasters; i++){
		curDist = euclideanDistance(image, clastersCenters[i]);
		if (curDist < minDist){
			minDist = curDist;
			clasterLabel = i;
		}
	}
	return clasterLabel;
}

void addImage(himage_t& image1, himage_t& image2){
	std::transform(image1.begin(), image1.end(), image2.begin(), image1.begin(),std::plus<float>());
}

void divImage(himage_t& image, float divider){
	std::transform(image.begin(), image.end(), image.begin(), [divider](float feature){return feature / divider; });
}

void fillImage(himage_t& image, float feature){
	std::fill(image.begin(), image.end(), feature);
}

void recalcClasterCenter(himage_set_t& images, hlabel_set_t& clastersLabels, himage_t& clasterCenter, label_t label){
	fillImage(clasterCenter, 0);

	int nImagesInClaster = 0;
	for (int i = 0; i < images.size(); i++){
		if (label == clastersLabels[i]){
			addImage(clasterCenter, images[i]);
			nImagesInClaster++;
		}
	}
	if (nImagesInClaster == 0)
		randomizeImage(clasterCenter);
	else
		divImage(clasterCenter, nImagesInClaster);
}

hlabel_set_t markImagesCPU(himage_set_t& images, int nClasters, int nIters = 10000){
	hlabel_set_t clastersLabels(images.size(), 0);
	
	int nImageFeatures = images[0].size();
	himage_set_t clastersCenters(nClasters);
	for (auto& center : clastersCenters)
		center.resize(nImageFeatures);

	himage_set_t prevClastersCenters = clastersCenters;
	
	for (auto& center : clastersCenters)
		randomizeImage(center);

	while (prevClastersCenters != clastersCenters){

		for (int i = 0; i < images.size(); i++)
			clastersLabels[i] = getClasterLabel(images[i], clastersCenters);

		prevClastersCenters = clastersCenters;
		
		for (int i = 0; i < nClasters; i++)
			recalcClasterCenter(images, clastersLabels, clastersCenters[i], i);
	}

	return clastersLabels;
}

ostream& outImagesPerEachClaster(hlabel_set_t& clastersLabels, int nClasters, ostream& s){
	s << "clasters number : " << nClasters <<endl;
	int nImagesInClaster = 0;
	int nImages = clastersLabels.size();
	for (int i = 0; i < nClasters; i++){
		nImagesInClaster = std::count_if(clastersLabels.begin(), clastersLabels.end(), [i](label_t& label){return label == i; });
		s << "claster " << i << " : " << nImagesInClaster << ", " << (float)nImagesInClaster / nImages << "%"<< endl;
	}
	return s;
}

__global__ void kern(float* raw_ptr, int size){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		raw_ptr[idx] += 1;
}

int main(int argc, char* argv[]){
	/*
	host_vector<float> hvec(2 << 21);
	thrust::generate(hvec.begin(), hvec.end(), rand);
	
	Timer t;
	
	//cudaMemcpy
	device_vector<float> dvec = hvec;
	float* dvec_raw_ptr = thrust::raw_pointer_cast(&dvec[0]);
	dim3 nThreads = dim3(1024);
	dim3 nBlocks = dim3((dvec.size() + nThreads.x - 1) / nThreads.x);
	t.start();
	cout << "threads : " << nThreads.x << endl;
	cout << "blocks : " << nBlocks.x << endl;
	kern <<< nBlocks, nThreads >>>(dvec_raw_ptr, dvec.size());
	cudaThreadSynchronize();
	double delta_device = t.time_diff();

	t.start();
	int size = hvec.size();
	for (int i = 0; i < size; i++)
		hvec[i] += 1;
	double delta_host = t.time_diff();

	cout << "device" << delta_device << endl;
	cout << "times : " << delta_host / delta_device << endl;
	host_vector<float> hvec1 = dvec;
	cout << "eq : " << (hvec1 == hvec);
	
	cout << "host : "<< endl;
	for (int i = 0; i < 10; i++)
		cout << hvec[i] << " ";

	cout << endl << "device : " << endl;
	for (int i = 0; i < 10; i++)
		cout << hvec1[i] << " ";
	*/
	
	int nClasters = num<int>(argv[2]);
	
	ifstream file(argv[1]);		
	himage_set_t images = imagesFromFile(file);
	file.close();

	if (images.empty()){
		std::system("pause");
		cout << "there is no images";
		exit(-1);
	}

	himage_set_t normImages = images;
	normalizeImages(normImages);
	
	Timer timer;
	timer.start();
	hlabel_set_t clastersLabels = markImagesCPU(normImages, nClasters);
	double time = timer.time_diff();
	
	cout << "features : " << images[0].size() << endl;
	outImagesPerEachClaster(clastersLabels, nClasters, cout);
	cout << "time : " << time << endl;
	
	std::system("pause");
	return 0;
}
