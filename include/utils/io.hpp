//
// Created by Bing LU on 25/5/26.
//

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>

namespace WALTZ {

template <typename T> T* readfile(const char* file, const size_t num) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cerr << " Error, Couldn't find the file: " << file << "\n";
        throw std::invalid_argument("Couldn't find the file");
    }
    fin.seekg(0, std::ios::end);
    if (fin.tellg() / sizeof(T) != num) {
        throw std::invalid_argument("File size is not equal to the input setting");
    }
    T* data = new T[num];
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char*>(data), num * sizeof(T));
    fin.close();
    return data;
}

// count is the number of T elements, not the number of bytes.
template <typename T>
void writefile(const char* path, const T* data, size_t count) {
    std::ofstream file(path, std::ios::binary);
    if (!file.write(reinterpret_cast<const char*>(data), count * sizeof(T)))
        throw std::runtime_error(std::string("Cannot write output: ") + path);
}

} // namespace WALTZ
