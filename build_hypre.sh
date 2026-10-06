#!/bin/bash

# bash build_and_test.sh [--cmake] [--make] [--notest]

set -e

# Control flags
RUN_CMAKE=false
RUN_MAKE=false
RUN_TESTS=true

# GPU selection: default to HIP to preserve previous behavior
GPU_TARGET="hip"   # valid: "hip" or "cuda"
GPU_ARCH="gfx1100"        # optional, passed to CMake as CMAKE_*_ARCHITECTURES

for arg in "$@"; do
    case $arg in
        --cmake)
            RUN_CMAKE=true
            RUN_MAKE=true
            ;;
        --make)
            RUN_MAKE=true
            ;;
        --notest)
            RUN_TESTS=false
            ;;
        --hip)
            GPU_TARGET="hip"
            ;;
        --cuda)
            GPU_TARGET="cuda"
            ;;
        --gpu=*)
            GPU_TARGET="${arg#--gpu=}"
            ;;
        --arch=*)
            GPU_ARCH="${arg#--arch=}"
            ;;
        *)
            ;;
    esac
done

mkdir -p build

STEP=1
echo "========================================"
echo "   Build and Test Script Summary"
echo "========================================"
echo "Steps to be executed:"
echo "  $STEP. Apply astyle formatting (src/)"; STEP=$((STEP + 1))
echo "  $STEP. Run MUP checks (src/)"; STEP=$((STEP + 1))
if [ "$RUN_CMAKE" = true ]; then
    echo "  $STEP. Run cmake with flag (build/)"; STEP=$((STEP + 1))
fi
if [ "$RUN_MAKE" = true ]; then
    echo "  $STEP. Run make -j 16 (build/)"; STEP=$((STEP + 1))
fi
if [ "$RUN_TESTS" = true ]; then
    echo "  $STEP. Run tests (testilu.sh)"; STEP=$((STEP + 1))
fi
echo "  GPU target: $GPU_TARGET${GPU_ARCH:+ (arch=$GPU_ARCH)}"
echo "========================================"
echo ""

# Apply astyle
cd src
echo "-------- [1/$(($STEP))] Applying astyle... --------"
bash config/astyle-apply.sh .
echo "-------- Astyle done --------"
echo ""

# MUP stage
echo "-------- [2/$(($STEP))] Running MUP checks... --------"
./mup_check && ./mup_code > /dev/null && ./mup_clean
echo "-------- MUP done --------"
echo ""
cd ..

# Build hypre
cd build
CURRENT_STEP=3
if [ "$RUN_CMAKE" = true ]; then
    echo "-------- [$CURRENT_STEP/$(($STEP))] Running cmake... --------"
    # Configure GPU-specific CMake flags depending on selection
    GPU_CMAKE_FLAGS=""
    if [ "$GPU_TARGET" = "cuda" ]; then
        GPU_CMAKE_FLAGS="-DHYPRE_ENABLE_CUDA=ON -DHYPRE_ENABLE_CALIPER=ON -DHYPRE_ENABLE_HIP=OFF -DHYPRE_ENABLE_ROCTHRUST=OFF -DHYPRE_ENABLE_THRUST_NOSYNC=ON"
        if [ -n "$GPU_ARCH" ]; then
            GPU_CMAKE_FLAGS="$GPU_CMAKE_FLAGS -DCMAKE_CUDA_ARCHITECTURES=$GPU_ARCH"
        fi
    else
        GPU_CMAKE_FLAGS="-DHYPRE_ENABLE_HIP=ON -DHYPRE_ENABLE_CUDA=OFF -DHYPRE_ENABLE_CALIPER=ON -DHYPRE_ENABLE_ROCTHRUST=ON -DHYPRE_ENABLE_THRUST_NOSYNC=ON"
        if [ -n "$GPU_ARCH" ]; then
            GPU_CMAKE_FLAGS="$GPU_CMAKE_FLAGS -DCMAKE_HIP_ARCHITECTURES=$GPU_ARCH"
        fi
    fi

    cmake ../src . \
        --fresh \
        -DHYPRE_ENABLE_MIXED_PRECISION=OFF \
        $GPU_CMAKE_FLAGS \
        -DHYPRE_ENABLE_UNIFIED_MEMORY=OFF \
        -DHYPRE_BUILD_TESTS=ON \
        -DHYPRE_BUILD_UMPIRE=ON \
        -DHYPRE_ENABLE_UMPIRE_HOST=ON \
        -DHYPRE_ENABLE_UMPIRE_PINNED=OFF \
        -DHYPRE_ENABLE_UMPIRE_DEVICE=ON
    echo "-------- cmake done --------"
    echo ""
    
    CURRENT_STEP=$((CURRENT_STEP + 1))
fi
if [ "$RUN_MAKE" = true ]; then
    echo "-------- [$CURRENT_STEP/$(($STEP))] Running make... --------"
    make -j 16
    make install
    cd _deps/umpire-build && cmake --install . && cd -
    echo "-------- make done --------"
    echo "" 
    CURRENT_STEP=$((CURRENT_STEP + 1))
fi
cd ..

# Run tests
if [ "$RUN_TESTS" = true ]; then
    echo "-------- [$CURRENT_STEP/$(($STEP))] Running tests... --------"
    cd src/test
    ./cleantest.sh
    echo "-------- Running testilu.sh --------"
    ./runtest.sh TEST_ij/ilu.sh
    echo "-------- Running checktest.sh --------"
    ./checktest.sh
    cd ../..
    echo "-------- Tests done --------"
fi
echo ""
echo "========================================"
echo "   All steps completed successfully!"
echo "========================================"
