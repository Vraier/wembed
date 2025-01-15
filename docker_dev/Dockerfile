FROM ubuntu:24.04

# install bash multiplexer and editor
RUN apt-get update && apt-get install -y --no-install-recommends bash tmux vim git
# otherwise git won't be able to clone repositories
RUN apt-get install --reinstall ca-certificates 

# c++ stuff and libraries
RUN apt-get install -y --no-install-recommends build-essential
RUN apt-get install -y --no-install-recommends cmake
RUN apt-get install -y --no-install-recommends libeigen3-dev
RUN apt-get install -y --no-install-recommends libboost-all-dev

# For animations
# RUN apt-get install -y --no-install-recommends libsfml-dev

# Python header
RUN apt-get install -y --no-install-recommends python3-dev