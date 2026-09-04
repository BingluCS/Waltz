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
#include <vector>

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
    } else if (fin.tellg() / sizeof(T) > num) {
        std::cout << "Warning, file size" << fin.tellg() / sizeof(T)
                  << " is larger than the input setting, only read " << num << " elements\n";
    }
    T* data = new T[num];
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char*>(data), num * sizeof(T));
    fin.close();
    return data;
}

template <typename T> void writefile(const char* file, T* data, size_t num_elements) {
    std::ofstream fout(file, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(&data[0]), num_elements * sizeof(T));
    fout.close();
}

template <typename T> void writeTextFile(const char* file, T* data, size_t num_elements) {
    std::ofstream fout(file);
    if (fout.is_open()) {
        std::copy_n(data, num_elements, std::ostream_iterator<T>(fout, "\n"));
        fout.close();
    } else {
        std::cerr << "Error, unable to open file for output: " << file << std::endl;
        throw std::invalid_argument("Couldn't open the file for output");
    }
}

// Read a binary host file and return a CUDA device allocation containing the
// same elements.  The caller owns the returned pointer and must cudaFree it.
template <typename T> T* readfile_d(const char* file, const size_t num) {
    std::ifstream fin(file, std::ios::binary);
    if (!fin) {
        std::cerr << " Error, Couldn't find the file: " << file << "\n";
        throw std::invalid_argument("Couldn't find the file");
    }

    fin.seekg(0, std::ios::end);
    const std::streamoff file_bytes = fin.tellg();
    if (file_bytes < 0)
        throw std::runtime_error("Failed to query file size");
    if (static_cast<size_t>(file_bytes) != num * sizeof(T))
        throw std::invalid_argument("File size is not equal to the input setting");

    std::vector<T> host_data(num);
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char*>(host_data.data()),
             static_cast<std::streamsize>(num * sizeof(T)));
    if (!fin)
        throw std::runtime_error("Failed to read input file");

    T* device_data = nullptr;
    cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&device_data), num * sizeof(T));
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    err = cudaMemcpy(device_data, host_data.data(), num * sizeof(T), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(device_data);
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return device_data;
}

// Copy a CUDA device allocation to the host and write it as a binary file.
template <typename T>
void writefile_d(const char* file, const T* device_data, size_t num_elements) {
    std::vector<T> host_data(num_elements);
    const cudaError_t err =
        cudaMemcpy(host_data.data(), device_data, num_elements * sizeof(T), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(err));

    std::ofstream fout(file, std::ios::binary);
    if (!fout) {
        std::cerr << "Error, unable to open file for output: " << file << "\n";
        throw std::invalid_argument("Couldn't open the file for output");
    }

    fout.write(reinterpret_cast<const char*>(host_data.data()),
               static_cast<std::streamsize>(num_elements * sizeof(T)));
    if (!fout)
        throw std::runtime_error("Failed to write output file");
}

} // namespace WALTZ
