DEBUGINFO=0
CC=g++

LAPACK_HOME=$(HOME)/lapack-3.7.1

CFLAGS=\
	-I$(HOME)/reps/googletest/googletest/include \
	-I$(LAPACK_HOME)/LAPACKE/include \
	-I/usr/local/include/eigen3 \
	-I$(HOME)/boost-inst/include \
	-I$(HOME)/eigen3 \
	-I/usr/local/include/boost \
	-O3 \
	-Wall -Wno-sign-compare 

# Makes act_even slower by 10-20%
ifeq ($(DEBUGINFO),1)
CFLAGS += -g
endif

LINKFLAGS=\
	  -L$(LAPACK_HOME) \
	  -L$(HOME)/reps/googletest/googletest/build \
	  -L$(HOME)/boost-inst/lib \
	  -L$(HOME)/lib64 \
	  -L/usr/local/lib/gcc/7 \
	  -lboost_random -lboost_program_options -lboost_iostreams \
	  -lboost_system -lboost_thread \
	  -llapacke -llapack -lblas -lgfortran

CUDA_CFLAGS=-I$(CUDA_HOME)/include
CUDA_LINKFLAGS=-L$(CUDA_HOME)/lib -lcudart -lcublas -lcusparse -lpthread

NVCC_FLAGS=-O3 -Xcompiler -Wall -Xcompiler -Wextra -arch=sm_35

HEADERS=$(wildcard src/*.h)
SRCS=$(wildcard src/*.cc)

OBJ_DIR=obj

OBJS:=\
    $(OBJ_DIR)/defs.o \
    $(OBJ_DIR)/eigen_utils.o \
    $(OBJ_DIR)/kitaev.o \
    $(OBJ_DIR)/BasisState.o \
    $(OBJ_DIR)/DisorderParameter.o \
    $(OBJ_DIR)/KitaevHamiltonian.o \
    $(OBJ_DIR)/KitaevHamiltonianBlock.o \
    $(OBJ_DIR)/Spectrum.o \
    $(OBJ_DIR)/MajoranaDisorderParameter.o \
    $(OBJ_DIR)/NaiveMajoranaKitaevHamiltonian.o \
    $(OBJ_DIR)/MajoranaKitaevHamiltonian.o \
    $(OBJ_DIR)/Correlators.o \
    $(OBJ_DIR)/Timer.o \
    $(OBJ_DIR)/FockSpaceUtils.o \
    $(OBJ_DIR)/RandomMatrix.o \
    $(OBJ_DIR)/TSVFile.o \
	$(OBJ_DIR)/FactorizedSpaceUtils.o \
	$(OBJ_DIR)/FactorizedHamiltonian.o \
	$(OBJ_DIR)/Lanczos.o

TEST_OBJS:=\
	$(OBJ_DIR)/TestUtils.o

CUDA_OBJS:=\
	$(OBJ_DIR)/CudaUtils.o \
	$(OBJ_DIR)/CudaState.o \
	$(OBJ_DIR)/CudaHamiltonian.o \
	$(OBJ_DIR)/CudaLanczos.o \
	$(OBJ_DIR)/CudaDeviceUtils.o \
	$(OBJ_DIR)/CudaMultiGpuHamiltonianNaive.o \
	$(OBJ_DIR)/CudaMultiGpuHamiltonian.o

BINS:=\
	test-kitaev \
	test-gpu \
	test-multi-gpu \
	kitaev \
	kitaev-thermodynamics \
	random-matrix \
	partition-function \
	partition-function-single-sample \
	partition-function-disorder-average \
	2pt-function-disorder-average \
	syk-low-energy-spectrum \
	syk-gpu-lanczos \
	syk-gpu-benchmark \
	lanczos-checkpoint-info \
	benchmark-test \
	compute_H_moment

all: mkdir $(BINS) TAGS GTAGS depend

TAGS: $(HEADERS) $(SRCS)
	etags -o TAGS src/*.cc src/*.h

# gtags doesn't work well on Sherlock, it gets
# stuck trying to recursively scan the dir tree,
# so we supply it with a fixed list of files.
GTAGS: $(HEADERS) $(SRCS)
	rm -f gtags-targets.txt
	ls src/*.cc src/*.h > gtags-targets.txt
	$(HOME)/bin/gtags -f gtags-targets.txt

clean:
	rm -f $(BINS) $(OBJ_DIR)/*.o

# I put -g in CFLAGS
#debug: CFLAGS += -DDEBUG -D_GLIBCXX_DEBUG -g
#debug: all

profile: CFLAGS += -pg
profile: LINKFLAGS += -pg
profile: benchmark-test

mkdir:
	mkdir -p $(OBJ_DIR)

### Dependencies ###

depend: $(SRCS) $(HEADERS)
	touch depend
	makedepend -f depend $(SRCS)
#makedepend -f depend $(SRCS) > /dev/null 2>&1

# Don't include other include dirs so it doesn't drag eigen in there
# (makes it fail on Sherlock)
#makedepend -f depend -- $(CFLAGS) -- $(SRCS) > /dev/null 2>&1

### Executables ###

test-kitaev: $(OBJ_DIR)/test-kitaev.o $(OBJ_DIR)/archived-tests.o $(OBJS) $(TEST_OBJS)
	$(CC) $^ $(LINKFLAGS) -lgtest -lpthread -o $@

kitaev: $(OBJ_DIR)/main.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

kitaev-thermodynamics: $(OBJ_DIR)/kitaev-thermodynamics.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

random-matrix: $(OBJ_DIR)/random-matrix.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

partition-function: $(OBJ_DIR)/partition-function.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

partition-function-single-sample: $(OBJ_DIR)/partition-function-single-sample.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

partition-function-disorder-average: $(OBJ_DIR)/partition-function-disorder-average.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

2pt-function-disorder-average: $(OBJ_DIR)/2pt-function-disorder-average.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

syk-low-energy-spectrum: $(OBJ_DIR)/syk-low-energy-spectrum.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

syk-gpu-lanczos: $(OBJ_DIR)/syk-gpu-lanczos.o $(OBJS) $(CUDA_OBJS)
	$(CC) $^ $(LINKFLAGS) $(CUDA_LINKFLAGS) -o $@ 

lanczos-checkpoint-info: $(OBJ_DIR)/lanczos-checkpoint-info.o $(OBJS) $(CUDA_OBJS)
	$(CC) $^ $(LINKFLAGS) $(CUDA_LINKFLAGS) -o $@ 

syk-gpu-benchmark: $(OBJ_DIR)/syk-gpu-benchmark.o $(OBJS) $(CUDA_OBJS)
	$(CC) $^ $(LINKFLAGS) $(CUDA_LINKFLAGS) -o $@ 

test-gpu: $(OBJ_DIR)/test-gpu.o $(OBJS) $(CUDA_OBJS) $(TEST_OBJS)
	$(CC) $^ $(LINKFLAGS) $(CUDA_LINKFLAGS) -lgtest -lpthread -o $@

test-multi-gpu: $(OBJ_DIR)/test-multi-gpu.o $(OBJS) $(CUDA_OBJS) $(TEST_OBJS)
	$(CC) $^ $(LINKFLAGS) $(CUDA_LINKFLAGS) -lgtest -lpthread -o $@

benchmark-test: $(OBJ_DIR)/benchmark-test.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

compute_H_moment: $(OBJ_DIR)/compute_H_moment.o $(OBJS)
	$(CC) $^ $(LINKFLAGS) -o $@

### Object files ###

$(OBJ_DIR)/CudaDeviceUtils.o: src/CudaDeviceUtils.cu
	nvcc -c src/CudaDeviceUtils.cu -o $@ $(NVCC_FLAGS)

$(OBJ_DIR)/%.o: src/%.cc
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) -c -o $@ $<

include depend
# DO NOT DELETE
