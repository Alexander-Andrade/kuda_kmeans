#include <cuda.h>
#include <cuda_runtime_api.h>
#include "Includes.h"
#include <functional>
#include <algorithm>
#include <utility>
#include <stdio.h>
#include <string.h>
#include <exception>

class CudaException : public exception{
	string e;
public:
	CudaException(const char* e){
		this->e = "Cuda Error : ";
		this->e.append(e);
	}
	virtual const char* what()const throw()
	{
		return this->e.data();
	}
};


using namespace std;

using himage_t = vector<float>;
using dimage_t = float*;
using himage_set_t = vector<himage_t>;
using dimage_set_t = dimage_t*;
using label_t = unsigned char;
using hlabel_set_t = vector<label_t>;
using uint = unsigned int;

//random engine
std::default_random_engine randGen((unsigned int)std::time(NULL));
std::uniform_real_distribution<float> unifDistr(0.0, 1.0);

int determFeatureNum(istream& stream, char delim = ','){
	long long cur_pos = stream.tellg();
	stream.seekg(0, stream.beg);

	string line;
	std::getline(stream, line);

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

void imagesFromFile(istream& stream, himage_set_t& images, int estimated_vec_num = 1024, char delim = ',', int n_ignored = 100){
	images.reserve(estimated_vec_num);
	int nImageFeatures = determFeatureNum(stream);
	float curFeature = 0;
	while (!stream.eof()){
		vector<float> curImage(nImageFeatures);
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
		images.push_back(curImage);
	}
}

std::pair<float, float> featureMinMax(himage_set_t& images, int featureNo){
	float min = images[0][featureNo];
	float max = min;
	float curFeature = 0;
	for (auto& image : images){
		curFeature = image[featureNo];
		if (curFeature > max)
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
	for (int i = 0; i < nImageFeatures; i++)
		normalizeFeature(images, i);
}

void randomizeImage(himage_t& image){
	for (auto& feature : image)
		feature = unifDistr(randGen);
}

void randomizeImage(float* image, int nImageFeatures){
	for (int i = 0; i < nImageFeatures;i++)
		image[i] = unifDistr(randGen);
}

double euclideanDistance(himage_t& image1, himage_t image2){
	double squaredDiffSum = 0.0;
	int nImageFeatures = image1.size();
	for (int i = 0; i < nImageFeatures; i++)
		squaredDiffSum += pow(image1[i] - image2[i], 2);
	return sqrt(squaredDiffSum);
}

__device__ float euclideanDistance(float* imgPtr1, float* imgPtr2, int nImageFeatures){
	float squaredDiffSum =0.0;
	for (int i = 0; i < nImageFeatures; i++)
		squaredDiffSum += powf(imgPtr1[i] - imgPtr2[i], 2);
	return sqrtf(squaredDiffSum);
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
	std::transform(image1.begin(), image1.end(), image2.begin(), image1.begin(), std::plus<float>());
}

__host__ __device__ void addImage(float* image1, float* image2, int nImageFeatures){
	for (int i = 0; i < nImageFeatures; i++){
		image1[i] += image2[i];
	}
}

void divImage(himage_t& image, float divider){
	std::transform(image.begin(), image.end(), image.begin(), [divider](float feature){return feature / divider; });
}

void divImage(float* image, int nImageFeatures, float divider){
	for (int i = 0; i < nImageFeatures; i++)
		image[i] /= divider;
}

void fillImage(himage_t& image, float feature){
	std::fill(image.begin(), image.end(), feature);
}

void recalcClasterCenter(himage_set_t& images, hlabel_set_t& clastersLabels, himage_t& clasterCenter, label_t label){
	fillImage(clasterCenter, 0);

	int nImagesInClaster = 0;
	for (size_t i = 0; i < images.size(); i++){
		if (label == clastersLabels[i]){
			addImage(clasterCenter, images[i]);
			nImagesInClaster++;
		}
	}
	if (nImagesInClaster == 0)
		randomizeImage(clasterCenter);
	else
		divImage(clasterCenter, (float)nImagesInClaster);
}


vector<int> countClasterImages(hlabel_set_t& clastersLabels, int nClasters){
	vector<int> clastersCounters(nClasters);
	for (int i = 0; i < nClasters;i++)
		clastersCounters[i] = std::count_if(clastersLabels.begin(), clastersLabels.end(), [i](label_t& label){return label == i; });
	return clastersCounters;
}

void checkCudaError(cudaError_t e){
	if (e != cudaSuccess)
		throw CudaException(cudaGetErrorString(e));
}

template<typename T>
T* linearPtrFromVectorMat(vector<vector<T>>& vec){
	int height = vec.size();
	int width = vec[0].size();

	T* mat = new T[height * width];
	int rowSize = width * sizeof(T);
	for (int i = 0; i < height; i++)
		memcpy(&mat[i * width], vec[i].data(), rowSize);
	return mat;
}

template<typename T>
T* allocDevice(int size){
	T* ptr = nullptr;
	cudaError_t allocError = cudaMalloc((void**)&ptr, size * sizeof(T));
	checkCudaError(allocError);
	return ptr;
}

template<typename T>
void cudaCopy(T* dst, T* src, int size, cudaMemcpyKind kind){
	cudaError_t copyError = cudaMemcpy(dst, src, size * sizeof(T), kind);
	checkCudaError(copyError);
}


hlabel_set_t markImagesCPU(himage_set_t& images, int nClasters){
	hlabel_set_t clastersLabels(images.size(), 0);

	int nImageFeatures = images[0].size();
	himage_set_t clastersCenters(nClasters);
	for (auto& center : clastersCenters)
		center.resize(nImageFeatures);

	himage_set_t prevClastersCenters = clastersCenters;

	for (auto& center : clastersCenters)
		randomizeImage(center);
	int nIters = 0;
	while (prevClastersCenters != clastersCenters){
		nIters++;
		for (size_t i = 0; i < images.size(); i++)
			clastersLabels[i] = getClasterLabel(images[i], clastersCenters);

		prevClastersCenters = clastersCenters;

		for (int i = 0; i < nClasters; i++)
			recalcClasterCenter(images, clastersLabels, clastersCenters[i], i);
	}
	cout << "nIters : " << nIters << endl;
	return clastersLabels;
}


__global__ void getClasterLabels(float* images, int nImages, float* clastersCenters, int nClasters, int nImageFeatures, label_t* clasterLabels){
	uint tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < nImages){
		float* imagePtr = images + tid * nImageFeatures;
		float minDist = euclideanDistance(imagePtr, clastersCenters, nImageFeatures);
		//printf("tid=%d %f %f %f %f\n", tid, imagePtr[0], imagePtr[1], imagePtr[2], imagePtr[3]);
		float curDist = 0.0;
		float clasterLabel = 0;
		for (int i = 1; i < nClasters; i++){
			curDist = euclideanDistance(imagePtr, &clastersCenters[i * nImageFeatures], nImageFeatures);
			if (curDist < minDist){
				minDist = curDist;
				clasterLabel = i;
			}
		}
		//printf("tid=%d, label=%f \n", tid, clasterLabel);
		clasterLabels[tid] = clasterLabel;
	}
}

template<typename T>
T* cudaClone2D(T* srcPtr, int height, int width, cudaMemcpyKind kind = cudaMemcpyHostToDevice, T* dstPtr = nullptr){
	if (dstPtr == nullptr)
		dstPtr = allocDevice<T>(height * width);
	cudaCopy(dstPtr, srcPtr, height * width, kind);
	return dstPtr;
}

void recalcClasterCenter(float* images, int nImages, label_t* clastersLabels, float* clasterCenter, int nImageFeatures, int label){
	memset(clasterCenter, 0, nImageFeatures * sizeof(float));
	
	int nImagesInClaster = 0;
	
	for (int i = 0; i < nImages; i++){
		if (label == clastersLabels[i]){
			addImage(clasterCenter, &images[i * nImageFeatures], nImageFeatures);
			nImagesInClaster++;
		}
	}
	if (nImagesInClaster == 0)
		randomizeImage(clasterCenter, nImageFeatures);
	else
		divImage(clasterCenter, nImageFeatures,(float)nImagesInClaster);
}


hlabel_set_t markImagesGPU(float* images, int nImages, int nImageFeatures, int nClasters){
	hlabel_set_t clastersLabels(nImages, 0);

	int nClastersFeatures = nClasters * nImageFeatures;
	int nbytesClastersFeatures = nClastersFeatures * sizeof(float);

	float* clastersCenters = new float[nClastersFeatures];
	float* prevClastersCenters = new float[nClastersFeatures];

	for (int i = 0; i < nClasters; i++)
		randomizeImage(&clastersCenters[i * nImageFeatures], nImageFeatures);

	float* dImages = cudaClone2D(images, nImages, nImageFeatures);
	label_t* dClastersLabels = allocDevice<label_t>(nImages);
	float* dClastersCenters = cudaClone2D(clastersCenters, nClasters, nImageFeatures);

	dim3 nThreads = dim3(1024);
	dim3 nBlocks = dim3((nImages + nThreads.x - 1) / nThreads.x);
	int nIters = 0;
	while (memcmp(clastersCenters, prevClastersCenters, nbytesClastersFeatures) != 0){
		//parallel labelig
		getClasterLabels<< <nBlocks, nThreads >> >(dImages, nImages, dClastersCenters, nClasters, nImageFeatures, dClastersLabels);
		cudaThreadSynchronize();
		cudaCopy(clastersLabels.data(), dClastersLabels, nImages, cudaMemcpyDeviceToHost);

		memcpy(prevClastersCenters, clastersCenters, nbytesClastersFeatures);

		for (int i = 0; i < nClasters; i++)
			recalcClasterCenter(images, nImages, clastersLabels.data(), &clastersCenters[i * nImageFeatures], nImageFeatures, i);
		
		cudaMemcpy(dClastersCenters, clastersCenters, nbytesClastersFeatures, cudaMemcpyHostToDevice);
		nIters++;
	}
	cout << "nIters : " << nIters << endl;

	cudaFreeHost(clastersCenters);
	delete[] prevClastersCenters;

	cudaFree(dClastersLabels);
	cudaFree(dClastersCenters);
	cudaFree(dImages);

	return clastersLabels;
}


int main(int argc, char* argv[]){
	int nClasters = num<int>(argv[2]);

	ifstream file(argv[1]);
	himage_set_t images;
	imagesFromFile(file, images);
	file.close();

	if (images.empty()){
		std::system("pause");
		cout << "there is no images";
		exit(-1);
	}

	himage_set_t normImages = images;
	normalizeImages(normImages);

	int nImages = images.size();
	int nImageFeatures = images[0].size();

	
	float* imagesPtr = linearPtrFromVectorMat(normImages);
	
	Timer timer;
	try{
		timer.start();
		//GPU algorithm
		hlabel_set_t gpuClastersLabels = markImagesGPU(imagesPtr, nImages, nImageFeatures, nClasters);
		
		double time = timer.time_diff();
		cout << "features : " << nImageFeatures << endl;
		vector<int> gpuClastCount = countClasterImages(gpuClastersLabels, nClasters);
		for (int i = 0; i < gpuClastCount.size(); i++)
			cout << "claster " << i << " : " << gpuClastCount[i] << endl;
		cout << "gpu time : " << time << endl;

		cudaFreeHost(imagesPtr);
	}
	catch(CudaException& e){
		cout << e.what();
	}
	
	cout << endl;

	timer.start();
	//CPU algorithm
	hlabel_set_t cpuClastersLabels = markImagesCPU(normImages, nClasters);
	
	double time = timer.time_diff();

	vector<int> cpuClastCount = countClasterImages(cpuClastersLabels, nClasters);
	for (int i = 0; i < cpuClastCount.size(); i++)
		cout << "claster " << i << " : " << cpuClastCount[i] << endl;
	cout << "cpu time : " << time << endl;
	

	std::system("pause");
	return 0;
}
