CXX = nvcc
CXXFLAGS = -O3 -std=c++14 -Wno-deprecated-gpu-targets -arch=sm_75

TARGETS = hello_cuda parallel_reduction image_blur exercise_2d_reduction exercise_convolution matrix_mul_demo

all: $(TARGETS)

hello_cuda: hello_cuda.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

parallel_reduction: parallel_reduction.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

image_blur: image_blur.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

exercise_2d_reduction: exercise_2d_reduction.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

exercise_convolution: exercise_convolution.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

matrix_mul_demo: matrix_mul_demo.cu
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)
