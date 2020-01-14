#include "json.hpp"    // https://github.com/nlohmann/json/tree/develop/single_include/nlohmann/json.hpp
#include <fstream>              // ifstream, ofstream
#include <iomanip>              // std::setw()
#include <iostream>
using json = nlohmann::json;
void main() {

	std::ifstream fin("config.json");   // 注意此处是相对路径
	json j;
	fin >> j;
	fin.close();

	// 检查读取的和预期的是否一致
	std::cout << j["name"] << std::endl;	// 
	std::cout << j["Camera_size"] << std::endl;	// 
	std::cout << j["M1"] << std::endl;	// 
	std::cout << j["D1"] << std::endl;	//
	std::cout << j["M2"] << std::endl;	// 
	std::cout << j["D2"] << std::endl;	// 

}
