# 

#!/bin/bash
set -e

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
ABI=$(python -c 'import tensorflow as tf; print(tf.sysconfig.CXX11_ABI_FLAG)')

echo "TF_CFLAGS: ${TF_CFLAGS[@]}"
echo "TF_LFLAGS: ${TF_LFLAGS[@]}"
echo "ABI: ${ABI}"

# Neighbors op
g++ -std=c++11 -shared tf_neighbors/tf_neighbors.cpp tf_neighbors/neighbors/neighbors.cpp cpp_utils/cloud/cloud.cpp \
-o tf_neighbors.so -fPIC \
"${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}" \
-O2 -D_GLIBCXX_USE_CXX11_ABI=${ABI}

g++ -std=c++11 -shared tf_neighbors/tf_batch_neighbors.cpp tf_neighbors/neighbors/neighbors.cpp cpp_utils/cloud/cloud.cpp \
-o tf_batch_neighbors.so -fPIC \
"${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}" \
-O2 -D_GLIBCXX_USE_CXX11_ABI=${ABI}

# Subsampling op
g++ -std=c++11 -shared tf_subsampling/tf_subsampling.cpp tf_subsampling/grid_subsampling/grid_subsampling.cpp cpp_utils/cloud/cloud.cpp \
-o tf_subsampling.so -fPIC \
"${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}" \
-O2 -D_GLIBCXX_USE_CXX11_ABI=${ABI}

g++ -std=c++11 -shared tf_subsampling/tf_batch_subsampling.cpp tf_subsampling/grid_subsampling/grid_subsampling.cpp cpp_utils/cloud/cloud.cpp \
-o tf_batch_subsampling.so -fPIC \
"${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}" \
-O2 -D_GLIBCXX_USE_CXX11_ABI=${ABI}