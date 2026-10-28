NVCC ?= nvcc
ROOT := $(abspath ..)
K_TILE ?= 256
ROPE_VECTORS ?= 192
SHARED_SOURCES := $(addprefix ../native_pi0/,tcgen_probe.cu nvfp4_probe.cu dag_probe.cu dag_matmul.cu norm.cu dag_scheduler.cuh attention.cu)
CXXFLAGS := -O3 -lineinfo -std=c++20 -arch=sm_100a \
 -DKITTENS_HOPPER -DKITTENS_BLACKWELL -DNATIVE_COMPUTE_CONSUMER_WARPS=4 \
 -DNATIVE_FP4_K_TILE=$(K_TILE) -DEDGE_ROPE_VECTORS_PER_TASK=$(ROPE_VECTORS) --expt-relaxed-constexpr --expt-extended-lambda \
 -I$(ROOT)/third_party/Megakernels/include \
 -I$(ROOT)/third_party/Megakernels/ThunderKittens/include
.PHONY: all clean FORCE
all: build/native.so
build/native.so: build/config native.cu $(wildcard *.cuh) $(SHARED_SOURCES)
	mkdir -p build
	$(NVCC) $(CXXFLAGS) -shared -Xcompiler -fPIC native.cu -lcuda -o $@
clean:
	rm -rf build

build/native_test.so: build/config native.cu $(wildcard *.cuh) tests/validate_ffn.py $(SHARED_SOURCES)
	mkdir -p build
	$(NVCC) $(CXXFLAGS) -DNATIVE_DAG_MATMUL_VALIDATION \
	 -I$(ROOT)/third_party/FlashRT/csrc -I$(ROOT)/third_party/cutlass/include \
	 -shared -Xcompiler -fPIC native.cu ../native_pi0/fp4_output_reference.cu \
	 ../native_pi0/launch_trace.cpp -lcupti -Xlinker -rpath -Xlinker /usr/local/cuda/lib64 -lcuda -o $@

# Rebuild on compiler/tile/flag changes rather than silently using a stale binary.
FORCE:
build/config: FORCE
	@mkdir -p build
	@echo '$(NVCC) $(CXXFLAGS)' > build/config.tmp
	@cmp -s build/config.tmp $@ || cp build/config.tmp $@
	@rm -f build/config.tmp

build/rope_reference.so: build/config tests/rope_reference.cu $(ROOT)/third_party/FlashRT/csrc/kernels/cosmos3_edge_misc.cu
	mkdir -p build
	$(NVCC) -O3 -std=c++20 -arch=sm_100a -DFLASHRT_HAVE_COSMOS3_EDGE \
	 -I$(ROOT)/third_party/FlashRT/csrc -shared -Xcompiler -fPIC \
	 tests/rope_reference.cu $(ROOT)/third_party/FlashRT/csrc/kernels/cosmos3_edge_misc.cu -o $@
