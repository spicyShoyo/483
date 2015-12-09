CXX = nvcc
CXXFLAGS = -arch sm_20
TESTFLAGS = -G -g
OBJS_FILES = checkReader.cu
PCA_FILES = checkReaderPCA.cu
all:
	$(CXX) $(CXXFLAGS) $(OBJS_FILES)

pca:
	$(CXX) $(CXXFLAGS) $(PCA_FILES)

debug:
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) $(OBJS_FILES)

clean:
	rm -rf $(EXE_BTREE) $(EXE_BTREE)-asan \
		$(EXE_RACER) \
		$(OBJS_DIR) \
		$(RESULTS_DIR)
