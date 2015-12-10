CXX = nvcc
CXXFLAGS = -arch sm_20
TESTFLAGS = -G -g
OBJS_FILES = src/checkReader.cu
PCA_FILES = src/checkReaderPCA.cu
TIME_FILES = src/checkReaderPCATime.cu

all:
	$(CXX) $(CXXFLAGS) $(PCA_FILES)


knn:
	$(CXX) $(CXXFLAGS) $(OBJS_FILES)


time:
	$(CXX) $(CXXFLAGS) $(TIME_FILES)


debug:
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) $(OBJS_FILES)


clean:
	rm ./a.out
