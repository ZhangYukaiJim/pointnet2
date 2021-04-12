# TF1.2
#g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# TF1.4
g++ -std=c++11 tf_interpolate.cpp -o tf_interpolate_so_hk.so -shared -fPIC -I /home/vgd/tensorflow/lib/python3.5/site-packages/tensorflow/include -I /usr/local/cuda-9.0/include -I /home/vgd/tensorflow/lib/python3.5/site-packages/tensorflow/include/external/nsync/public -lcudart -L /usr/local/cuda-9.0/lib64/ -L /home/vgd/tensorflow/lib/python3.5/site-packages/tensorflow -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
# TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
# TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
# NSYNC_INC=$TF_INC/external/nsync/public
# g++-5 -std=c++11 -shared -o tf_interpolate_so_hk.so tf_interpolate.cpp -fPIC -I $TF_INC -I$NSYNC_INC -L$TF_LIB -L/usr/local/cuda/targets/x86_64-linux/lib -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
