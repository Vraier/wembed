FROM quay.io/pypa/manylinux_2_28_x86_64

# install bash multiplexer and editor
RUN yum update -y && yum install -y bash tmux vim git wget

# c++ stuff and libraries
RUN yum groupinstall -y "Development Tools"
RUN yum install -y cmake

# Python header
RUN yum install -y python3-devel

# install boost 
WORKDIR /
RUN wget https://archives.boost.io/release/1.82.0/source/boost_1_82_0.tar.gz
RUN tar -xvzf boost_1_82_0.tar.gz
RUN cp -r boost_1_82_0/boost /usr/local/include

# install eigen3
WORKDIR /
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
RUN tar -xvzf eigen-3.4.0.tar.gz
WORKDIR /eigen-3.4.0
RUN mkdir build
WORKDIR /eigen-3.4.0/build
RUN cmake ..
RUN make install