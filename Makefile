CXX = nvcc
CXXFLAGS = -arch sm_20
TESTFLAGS = -G -g
OBJS_FILES = checkReaderv2.cu

all:
	$(CXX) $(CXXFLAGS) $(OBJS_FILES)

t:
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) $(OBJS_FILES)