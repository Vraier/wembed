FROM ubuntu:24.04

# install bash multiplexer and editor
RUN apt-get update && apt-get install -y --no-install-recommends bash tmux vim git curl ca-certificates

# c++ stuff and libraries
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends cmake
RUN apt-get install -y --no-install-recommends libeigen3-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev

# Python header
RUN apt-get install -y --no-install-recommends python3-dev

# Cargo
RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
RUN echo 'source $HOME/.cargo/env' >> $HOME/.bashrc
