CXX = nvcc
CXXFLAGS = -arch sm_20
TESTFLAGS = -G -g
OBJS_FILES = checkReader.cu
PCA_FILES = checkReaderPCA.cu
non_pca:
	$(CXX) $(CXXFLAGS) $(OBJS_FILES)

pca:
	$(CXX) $(CXXFLAGS) $(PCA_FILES)

debug:
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) $(OBJS_FILES)

clean:
	rm ./a.out
