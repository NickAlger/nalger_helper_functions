default: all

##################    VVVV    Change these    VVVV    ##################

EIGEN_INCLUDE := /home/nick/anaconda3/envs/fenics3/include/eigen3 # https://eigen.tuxfamily.org/index.php?title=Main_Page
THREADPOOL_INCLUDE := /home/nick/repos/thread-pool # https://github.com/bshoshany/thread-pool

########################################################################

PYFLAGS  = $(shell python3 -m pybind11 --includes)
PYSUFFIX = $(shell python3-config --extension-suffix)

INCLUDE_DIR := ./include
SRC_DIR  := ./src
OBJ_DIR  := ./obj
BUILD_DIR  := ./nalger_helper_functions
# LIB_DIR  := ./lib
LIB_DIR  := $(BUILD_DIR)
EXAMPLES_DIR := ./examples

CXXFLAGS := -std=c++17 -pthread -lpthread -O3 -Wall
SHAREDFLAGS := -shared -fPIC

ALL_COMPILE_STUFF = $(CXXFLAGS) $(PYFLAGS) \
					-I$(INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(THREADPOOL_INCLUDE)

BINDINGS_TARGET = nalger_helper_functions_cpp.so

all: $(LIB_DIR)/$(BINDINGS_TARGET) $(EXAMPLES_DIR)/kdtree_example $(EXAMPLES_DIR)/aabbtree_example
	@echo 'Finished building target: $@'
	@echo ' '

$(LIB_DIR)/$(BINDINGS_TARGET): $(SRC_DIR)/pybind11_bindings.cpp $(INCLUDE_DIR)/kdtree.h $(INCLUDE_DIR)/aabbtree.h $(INCLUDE_DIR)/simplexmesh.h | $(LIB_DIR)
	@echo 'Building target: $@'
	g++ -o "$@" "$<" $(CXXFLAGS) $(SHAREDFLAGS) $(PYFLAGS) -I$(INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(THREADPOOL_INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

$(EXAMPLES_DIR)/kdtree_example: $(EXAMPLES_DIR)/kdtree_example.cpp $(INCLUDE_DIR)/kdtree.h
	@echo 'Building target: $@'
	g++ -o "$@" "$<" $(CXXFLAGS) -I$(INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(THREADPOOL_INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

$(EXAMPLES_DIR)/aabbtree_example: $(EXAMPLES_DIR)/aabbtree_example.cpp $(INCLUDE_DIR)/aabbtree.h
	@echo 'Building target: $@'
	g++ -o "$@" "$<" $(CXXFLAGS) -I$(INCLUDE_DIR) -I$(EIGEN_INCLUDE) -I$(THREADPOOL_INCLUDE)
	@echo 'Finished building target: $@'
	@echo ' '

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

clean:
	-rm -rf $(EXAMPLES_DIR)/kdtree_example
	-rm -rf $(EXAMPLES_DIR)/aabbtree_example
	-rm -rf $(LIB_DIR)/$(BINDINGS_TARGET)
	-@echo ' '

.PHONY: all clean dependents
