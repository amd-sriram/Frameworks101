#include "utils.hpp"
#include <chrono>
#include <random>

__global__ void add_kernel(double* a, double* b, double* result, int n){

	int tid=threadIdx.x+ blockIdx.x * blockDim.x;

	if(tid < n){
	   result[tid]= a[tid] + b[tid];
	}
}

__host__ bool check_result(const std::vector<double>& result, const std::vector<double>& A, std::vector<double>& B){

	for(int i=0; i<result.size(); i++) {
	   if(result[i] != A[i]+B[i]) {
	      return false;
	   }
	}

	return true;
}

int main() {

    int device_count;
    HIP_CHECK(hipGetDeviceCount(&device_count));

    std::cout  << "device_count: " << device_count << '\n';

    for(int i = 0; i < device_count; i++)
    {
        HIP_CHECK(hipSetDevice(i));
        // Query the device properties.
        hipDeviceProp_t props{};
        HIP_CHECK(hipGetDeviceProperties(&props, i));
        std::cout  << "Name: " << props.name << '\n';

        const int n=100000;

        int numBlocks = n/props.maxThreadsPerBlock;
        if (n % props.maxThreadsPerBlock) numBlocks++;
        int threadsPerBlock =  props.maxThreadsPerBlock;

        std::cout << "NumBlocks: " << numBlocks << " NumThreads: " << threadsPerBlock << "\n";

        // host array A, B and results array C
        std::vector<double> hostA(n);
        std::vector<double> hostB(n);
        std::vector<double> hostC(n);

        std::mt19937_64 rng;
        // initialize the random number generator with time-dependent seed
        uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
        rng.seed(ss);
        // initialize a uniform distribution between 0 and 1
        std::uniform_real_distribution<double> unif(0, 1);
        // host array init with uniform random numbers between 0 and 1
        for(int i=0; i<n; i++){
	        hostA[i] = unif(rng);
	        hostB[i] = unif(rng);
        }

        // Allocate device arrays
        double* deviceA;
        double* deviceB;
        double* deviceC;

        HIP_CHECK(hipMalloc(&deviceA, n*sizeof(double)));
        HIP_CHECK(hipMalloc(&deviceB, n*sizeof(double)));
        HIP_CHECK(hipMalloc(&deviceC, n*sizeof(double)));

        HIP_CHECK(hipMemcpy(deviceA, hostA.data(), n*sizeof(double), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(deviceB, hostB.data(), n*sizeof(double), hipMemcpyHostToDevice));


        add_kernel<<< dim3(numBlocks), dim3(threadsPerBlock), 0, hipStreamDefault >>>(deviceA, deviceB, deviceC, n);

        HIP_CHECK(hipDeviceSynchronize());


        HIP_CHECK(hipMemcpy(hostC.data(), deviceC, n*sizeof(double), hipMemcpyDeviceToHost));


        if(check_result(hostC, hostA, hostB)){
            std::cout << "result match between host and device" << std::endl;
        }
        else{
            std::cout << "result do not match between host and device"<< std::endl;
        }

        HIP_CHECK(hipFree(deviceA));
        HIP_CHECK(hipFree(deviceB));
        HIP_CHECK(hipFree(deviceC));
    }
    return 0;
}