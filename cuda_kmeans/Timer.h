#ifndef TIMER_H
#define TIMER_H
#include "Includes.h"

typedef long long int64_t;
typedef unsigned long long uint64_t;


class Timer{
private:
	uint64_t _t1;
	uint64_t _freq;
public:
	Timer(){
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		_freq = freq.QuadPart;
	}
	void nano_start(){
		_t1 = __rdtsc();
	}
	void start(){
		LARGE_INTEGER time_stemp;
		QueryPerformanceCounter(&time_stemp);
		_t1 = time_stemp.QuadPart;
	}
	double nano_time_diff(){ return (double)(__rdtsc() - _t1) / _freq / 1000; }
	double time_diff(){
		LARGE_INTEGER time_stemp;
		QueryPerformanceCounter(&time_stemp);
		return (double)(time_stemp.QuadPart - _t1) / _freq;
	}
};


#endif //TIMER_H