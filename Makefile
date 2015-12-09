CXX = nvcc
CXXFLAGS = -arch sm_20
TESTFLAGS = -G -g
OBJS_FILES = checkReader.cu
PCA_FILES = checkReaderPCA.cu

all:
	$(CXX) $(CXXFLAGS) $(PCA_FILES)


knn:
	$(CXX) $(CXXFLAGS) $(OBJS_FILES)


debug:
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) $(OBJS_FILES)


clean:
	rm ./a.out
