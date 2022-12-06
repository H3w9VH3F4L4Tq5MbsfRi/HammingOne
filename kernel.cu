
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <stdio.h>
#include <fstream>
#include <string>
#include <iostream>
#include <bitset>

#define size 64
#define threadsPerBlock 1024
#define megaToNormal 1000000

cudaError_t calcWithCuda(const unsigned long long* data, const int n, const int rSize, const bool verbose);

__global__ void addKernel(const unsigned long long* data, const int n, const int rSize, const bool verbose, unsigned long long* pairs)
{
    int i = blockIdx.x * threadsPerBlock + threadIdx.x;
    if (i >= n)
        return;

    unsigned long long pairsCounter = 0;

    for (int j = 0; j < n; j++)
    {
        if (j == i)
            continue;

        bool singleDiffSpotted = false;
        bool multipleDiffSpotted = false;
        for (int k = 0; k < rSize; k++)
        {
            if (multipleDiffSpotted)
                break;
            unsigned long long xored = data[i + n * k] ^ data[j + n * k];
            if (xored == 0)
                continue;
            unsigned long long testValue = xored & (xored - 1);
            if (testValue == 0 && !singleDiffSpotted)
            {
                singleDiffSpotted = true;
                continue;
            }
            multipleDiffSpotted = true;
        }

        if (!multipleDiffSpotted && singleDiffSpotted)
        {
            if (verbose)
                printf("%d is a pair with %d \n", i, j);
            pairsCounter++;
        }
    }
    pairs[i] = pairsCounter;

    return;
}

int main(int argc, char** argv)
{
    std::cout << "Patryk Saj" << std::endl;
    std::cout << "GPU Project 1" << std::endl;
    std::cout << "Hamming one" << std::endl << std::endl;

    if (argc < 2 || argc > 4)
    {
        std::cout << "Invalid parameters!" << std::endl;
        std::cout << "Terminating program..." << std::endl;
        return 1;
    }

    //Testing provided file for expected dataformat
    std::string path = argv[1];
    int n;
    int l;
    try
    {
        std::ifstream file(path);
        std::string str;
        std::getline(file, str);
        std::string nStr = str.substr(0, str.find(','));
        n = stoi(nStr);
        std::string lStr = str.substr(str.find(',') + 1, str.length());
        l = stoi(lStr);
    }
    catch (...)
    {
        std::cout << "Unable to read provided file!" << std::endl;
        std::cout << "Terminating program..." << std::endl;
        return 1;
    }

    // Reading data
    std::cout << "Reading from " << path << std::endl;
    std::cout << "n = " << n << std::endl << "l = " << l << std::endl;

    int rSize = (int)ceil((double)l / size);
    unsigned long long* data = new unsigned long long[rSize * n];
    std::ifstream file(path);
    std::string str;
    std::string str2;
    std::getline(file, str);

    auto start0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n; i++)
    {
        std::getline(file, str);
        for (int j = 0; j < rSize; j++)
        {
            str2 = str.substr(j * size, size);
            data[i + n * j] = std::bitset<size>(str2).to_ullong();
        }
    }
    auto stop0 = std::chrono::high_resolution_clock::now();
    auto duration0 = std::chrono::duration_cast<std::chrono::microseconds>(stop0 - start0);
    std::cout << "Data loading took " << duration0.count() / (double)megaToNormal << " s." << std::endl;

    //Getting parameters
    bool CPUversion = false;
    bool verbose = false;
    for (int i = 2; i < 4; i++)
    {
        if (argc > i)
        {
            std::string param = argv[i];

            if (param[0] == '-')
            {
                if (param[1] == 'v' && verbose == false)
                    verbose = true;
                else if (param[1] == 'c' && CPUversion == false)
                    CPUversion = true;
                else
                {
                    std::cout << "Invalid parameters!" << std::endl;
                    std::cout << "Terminating program..." << std::endl;
                    return 1;
                }
            }
            else
            {
                std::cout << "Invalid parameters!" << std::endl;
                std::cout << "Terminating program..." << std::endl;
                return 1;
            }
        }
    }

    if (CPUversion)
    {
        // CPU version of the algorithm
        std::cout << std::endl << "CPU version of the algorythm:" << std::endl;

        unsigned long long pairsCounter = 0;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (j == i)
                    continue;

                bool singleDiffSpotted = false;
                bool multipleDiffSpotted = false;
                for (int k = 0; k < rSize; k++)
                {
                    if (multipleDiffSpotted)
                        break;
                    unsigned long long xored = data[i + n * k] ^ data[j + n * k];
                    if (xored == 0)
                        continue;
                    unsigned long long testValue = xored & (xored - 1);
                    if (testValue == 0 && !singleDiffSpotted)
                    {
                        singleDiffSpotted = true;
                        continue;
                    }
                    multipleDiffSpotted = true;
                }

                if (!multipleDiffSpotted && singleDiffSpotted)
                {
                    if (verbose)
                        std::cout << i << " is a pair with " << j << std::endl;
                    pairsCounter++;
                }
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "CPU version execution time: " << duration.count() / (double)megaToNormal << " s" << std::endl;
        std::cout << "Found " << pairsCounter / 2 << " pairs." << std::endl;
    }

    //GPU version of the algorithm
    std::cout << std::endl << "GPU version of the algorythm:" << std::endl;
    cudaError_t cudaStatus = calcWithCuda(data, n, rSize, verbose);

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

cudaError_t calcWithCuda(const unsigned long long* data, const int n, const int rSize, const bool verbose)
{
    unsigned long long* dev_data = 0;
    unsigned long long* dev_pairs = 0;
    cudaError_t cudaStatus;
    int count = (int)ceil((double)n / threadsPerBlock);

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_data, n * rSize * sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_pairs, n * sizeof(unsigned long long));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_data, data, n * rSize * sizeof(unsigned long long), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    auto start2 = std::chrono::high_resolution_clock::now();
    addKernel <<< count, threadsPerBlock >>> (dev_data, n, rSize, verbose, dev_pairs);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    auto stop2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start2);
    std::cout << "GPU version execution time: " << duration2.count() / (double)megaToNormal << " s" << std::endl;

    unsigned long long* GPUpairs = new unsigned long long[n];
    cudaStatus = cudaMemcpy(GPUpairs, dev_pairs, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    unsigned long long counter = 0;
    for (int i = 0; i < n; i++)
        counter += GPUpairs[i];
    std::cout << "Found " << counter / 2 << " pairs." << std::endl;


Error:
    cudaFree(dev_data);
    cudaFree(dev_pairs);

    return cudaStatus;
}
