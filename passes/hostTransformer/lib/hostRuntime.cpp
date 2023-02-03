#include <fstream>
#include <sstream>
#include <cstdint>
#include <cassert>
#include <sys/time.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include "pathToConditionAnalyzer.h"

extern "C" __attribute__((weak))
void logKernelLaunchArguments(uint8_t* kernelName,
                              uint64_t gridXY,
                              uint32_t gridZ,
                              uint64_t blockXY,
                              uint32_t blockZ,
                              uint8_t** kernelArgs,
                              uint32_t* kernelArgIdx,
                              uint32_t numIdx,
                              bool dependsOnIndvar,
                              int* cnt)
{
    if (dependsOnIndvar || (*cnt) == 0)
    {
        std::ofstream out_file("host_param.out");
        std::ofstream out_file_all("host_param_all.out", std::ios::app);
        assert (out_file.is_open() && "Cannot open output file!\n");
        assert (out_file_all.is_open() && "Cannot open config log file!\n");

	std::stringstream config;
        config << kernelName << ','
	       << gridXY << ',' << gridZ << ',' << blockXY << ',' << blockZ;
        for (uint32_t i=0; i<numIdx; ++i)
        {
            uint32_t idx = kernelArgIdx[i];
            uint32_t arg = *(reinterpret_cast<uint32_t*>(kernelArgs[idx]));
            config << ",[" << idx << ']' << arg;
        }
        config << ",\n";

	out_file << config.str();
	out_file_all << config.str();
    }
}


extern "C" struct timeval *__attribute__((weak)) startClock() {
  struct timeval *t = new struct timeval();
  gettimeofday(t, nullptr);
  return t;
}


extern "C" void __attribute__((weak)) logElapsedTime(uint8_t* kernelName,
                                                     timeval* t1,
                                                     timeval* t2,
                                                     bool dependsOnIndvar,
                                                     int* cnt)
{
    if (dependsOnIndvar || (*cnt) == 0)
    {
        unsigned long tElapsed = (t2->tv_sec - t1->tv_sec) * 10e6 + (t2->tv_usec - t1->tv_usec);

        std::ofstream out_file("log_time.out", std::ios::out | std::ios::app );
        assert (out_file.is_open() && "Cannot open output file!\n");

        out_file << kernelName << ',' << tElapsed << '\n';
    }
    free(t1);
    free(t2);
}


extern "C" void __attribute__((weak)) kernelElapsedTime(uint8_t* kernelName,
                                                        timeval* t1)
{
    cudaDeviceSynchronize();
    struct timeval t2;
    gettimeofday(&t2, nullptr);
    unsigned long tElapsed = (t2.tv_sec - t1->tv_sec) * 10e6 + (t2.tv_usec - t1->tv_usec);
    free(t1);

    std::ofstream out_file("kernel_time.out", std::ios::out | std::ios::app );
    assert (out_file.is_open() && "Cannot open output file!\n");

    out_file << kernelName << ',' << tElapsed << '\n';
}


extern "C" void __attribute__((weak))
callConditionAnalyzerScript(uint8_t* kernelName, bool dependsOnIndvar, int* cnt)
{
    if (dependsOnIndvar || *cnt == 0)
    {
        const std::string kernelNameStr ((char *) kernelName);
        const std::string command = "python3 ";
        const std::string filename = "analysis_main.py ";
        const std::string args = "-d " + kernelNameStr + ".out " + " -p host_param.out";
        const std::string total_cmd = command + path + '/' + filename + args;
        std::system(total_cmd.c_str());
    }
    ++(*cnt);
}
