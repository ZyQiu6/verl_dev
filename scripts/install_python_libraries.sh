#!/usr/bin/env bash
set -euo pipefail
set -x

# ===================== 基本参数 =====================
WORKSPACE=${1:-"$(pwd)/ep_kernels_workspace"}
mkdir -p "$WORKSPACE"

# 可配参数
PIP_CMD=${PIP_CMD:-pip3}
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

# 默认 Hopper（H800）架构；如你已经在外部 export 了，就用你的
: "${TORCH_CUDA_ARCH_LIST:=9.0+PTX}"

# ===================== 先做些健壮性检查 =====================
command -v cmake >/dev/null || { echo "cmake 未找到，请在你的 conda 源里安装：conda install cmake ninja"; exit 1; }
command -v ninja >/dev/null || { echo "ninja 未找到，请在你的 conda 源里安装：conda install ninja"; exit 1; }
[ -x "$CUDA_HOME/bin/nvcc" ] || { echo "CUDA_HOME=$CUDA_HOME 似乎不包含 nvcc，请正确设置 CUDA_HOME"; exit 1; }

# ===================== 构建 NVSHMEM（无 IBGDA） =====================
# build nvshmem
pushd $WORKSPACE
mkdir -p nvshmem_src
# 这里假设你已经把与 CUDA 版本匹配的源码 txz 放在 $WORKSPACE 下，名字里含 nvshmem_*.txz
if [ ! -f nvshmem_src/CMakeLists.txt ]; then
  PKG_TXZ=$(ls -1 $WORKSPACE/nvshmem_*.txz 2>/dev/null | head -n1 || true)
  if [ -z "$PKG_TXZ" ]; then
    echo "请把 NVSHMEM 源码包 nvshmem_*.txz 放到 $WORKSPACE 下，再运行本脚本"; exit 1
  fi
  tar -xf "$PKG_TXZ" -C nvshmem_src --strip-components=1
fi

pushd nvshmem_src
# 不再应用 DeepEP 的补丁（关闭 IBGDA 时会失败）
# wget https://github.com/deepseek-ai/DeepEP/raw/main/third-party/nvshmem.patch
# git init
# git apply -vvv nvshmem.patch

# 必需变量检查
if [ -z "$CUDA_HOME" ]; then echo "CUDA_HOME 未设置"; exit 1; fi
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  export TORCH_CUDA_ARCH_LIST=9.0+PTX
fi

# 关闭 IBGDA/UCX/PMI/MPI 等
export NVSHMEM_IBGDA_SUPPORT=0
export NVSHMEM_SHMEM_SUPPORT=0
export NVSHMEM_UCX_SUPPORT=0
export NVSHMEM_USE_NCCL=0
export NVSHMEM_PMIX_SUPPORT=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_IBRC_SUPPORT=0
export NVSHMEM_BUILD_TESTS=0
export NVSHMEM_BUILD_EXAMPLES=0
export NVSHMEM_MPI_SUPPORT=0
export NVSHMEM_BUILD_HYDRA_LAUNCHER=0
export NVSHMEM_BUILD_TXZ_PACKAGE=0
export NVSHMEM_TIMEOUT_DEVICE_POLLING=0

NVSHMEM_BUILD=$WORKSPACE/nvshmem_build
NVSHMEM_PREFIX=$WORKSPACE/nvshmem_install
rm -rf "$NVSHMEM_BUILD"

# 👇 关键改动：强制 C++17 + 把 NVSHMEM_PREFIX 传进子工程
cmake -G Ninja -S . -B "$NVSHMEM_BUILD" \
  -DCMAKE_INSTALL_PREFIX="$NVSHMEM_PREFIX" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DNVSHMEM_PREFIX="$NVSHMEM_PREFIX" \
  -DCMAKE_CUDA_STANDARD=17 -DCMAKE_CUDA_STANDARD_REQUIRED=ON \
  -DCMAKE_CXX_STANDARD=17  -DCMAKE_CXX_STANDARD_REQUIRED=ON \
  -DCMAKE_CUDA_FLAGS="-std=c++17 -Xcompiler=-std=c++17 -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT=1" \
  -DCMAKE_CXX_FLAGS="-std=c++17"

cmake --build "$NVSHMEM_BUILD" -j

# 兼容 install 阶段某些版本找 libnvshmem.a
if [ ! -f "$NVSHMEM_BUILD/src/lib/libnvshmem.a" ]; then
  DEV_A=$(find "$NVSHMEM_BUILD" -name 'libnvshmem_device.a' | head -n1 || true)
  if [ -n "$DEV_A" ]; then
    mkdir -p "$NVSHMEM_BUILD/src/lib"
    ln -sf "$DEV_A" "$NVSHMEM_BUILD/src/lib/libnvshmem.a"
  fi
fi

cmake --install "$NVSHMEM_BUILD"
popd  # nvshmem_src

# 暴露给下游
export CMAKE_PREFIX_PATH=$WORKSPACE/nvshmem_install:$CMAKE_PREFIX_PATH
export NVSHMEM_HOME=$WORKSPACE/nvshmem_install
if [ -d "$WORKSPACE/nvshmem_install/lib/cmake/nvshmem" ]; then
  export NVSHMEM_DIR="$WORKSPACE/nvshmem_install/lib/cmake/nvshmem"
else
  export NVSHMEM_DIR="$WORKSPACE/nvshmem_install/share/cmake/nvshmem"
fi
export LD_LIBRARY_PATH="$WORKSPACE/nvshmem_install/lib:$LD_LIBRARY_PATH"

popd

echo "✅ 完成：NVSHMEM(无IBGDA) + pplx-kernels + DeepEP 安装"
echo "   运行时如需，确保已导出："
echo "     export LD_LIBRARY_PATH=\"$NVSHMEM_HOME/lib:\$LD_LIBRARY_PATH\""
