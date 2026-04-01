FROM ros:humble

ENV DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    wget \
    unzip \
    git \
    libgoogle-glog-dev \
    libatlas-base-dev \
    libsuitesparse-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-python-dev \
    ros-humble-image-transport \
    ros-humble-cv-bridge \
    ros-humble-vision-opencv \
    && rm -rf /var/lib/apt/lists/*

# Build and install Eigen 3.4.0
RUN wget -O /tmp/eigen-3.4.0.zip https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip \
    && unzip /tmp/eigen-3.4.0.zip -d /tmp \
    && cmake -S /tmp/eigen-3.4.0 -B /tmp/eigen-build \
    && cmake --build /tmp/eigen-build --target install \
    && rm -rf /tmp/eigen-3.4.0 /tmp/eigen-build /tmp/eigen-3.4.0.zip

# Build and install OpenCV 4.8.0
RUN git clone https://github.com/opencv/opencv.git -b 4.8.0 --depth 1 /tmp/opencv \
    && git clone https://github.com/opencv/opencv_contrib.git -b 4.8.0 --depth 1 /tmp/opencv_contrib \
    && cmake \
        -S /tmp/opencv \
        -B /tmp/opencv-build \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv_contrib/modules \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D WITH_IPP=OFF \
        -D WITH_GSTREAMER=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=OFF \
        -D BUILD_opencv_python2=OFF \
        -D BUILD_opencv_dnn=OFF \
        -D BUILD_opencv_ml=OFF \
        -D BUILD_opencv_stitching=OFF \
        -D BUILD_opencv_photo=OFF \
    && cmake --build /tmp/opencv-build -j$(nproc) \
    && cmake --install /tmp/opencv-build \
    && ldconfig \
    && rm -rf /tmp/opencv /tmp/opencv_contrib /tmp/opencv-build

# Build and install Ceres Solver 2.1.0
RUN wget -O /tmp/ceres-solver-2.1.0.tar.gz http://ceres-solver.org/ceres-solver-2.1.0.tar.gz \
    && tar zxf /tmp/ceres-solver-2.1.0.tar.gz -C /tmp \
    && cmake \
        -S /tmp/ceres-solver-2.1.0 \
        -B /tmp/ceres-build \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D EXPORT_BUILD_DIR=ON \
    && cmake --build /tmp/ceres-build -j$(nproc) \
    && cmake --install /tmp/ceres-build \
    && rm -rf /tmp/ceres-solver-2.1.0 /tmp/ceres-build /tmp/ceres-solver-2.1.0.tar.gz

# Set up workspace
ENV WORKSPACE=/workspace
WORKDIR $WORKSPACE

# Source ROS2 on every bash session
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "if [ -f $WORKSPACE/install/setup.bash ]; then source $WORKSPACE/install/setup.bash; fi" >> /root/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && exec \"$@\"", "--"]
CMD ["bash"]
