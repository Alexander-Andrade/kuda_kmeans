#ifndef STR_NUM
#define STR_NUM

#include <sstream>
#include <string>

using namespace std;

template<typename T>
std::string str(T num){
	ostringstream str_stream;
	str_stream << num;
	return str_stream.str();
}

template<typename T>
T num(std::string& str){
	T num = 0;
	istringstream(str.c_str()) >> num;
	return num;
}

template<typename T>
T num(char* str){
	T num = 0;
	istringstream(str) >> num;
	return num;
}


#endif