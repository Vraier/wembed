FROM ubuntu:24.04

# install bash multiplexer and editor
RUN apt-get update && apt-get install -y --no-install-recommends bash tmux vim git

# c++ stuff and libraries
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends cmake
RUN apt-get install -y --no-install-recommends libeigen3-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev

# Python header
RUN apt-get install -y --no-install-recommends python3-dev