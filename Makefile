CXX = nvcc
CXXFLAGS = -arch sm_20
TESTFLAGS = -G -g
OBJS_FILES = checkReader.cu
PCA_FILES = checkReaderPCA.cu
TIME_FILES = checkReaderPCATime.cu

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
