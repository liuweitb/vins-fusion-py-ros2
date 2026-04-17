# Python VINS-Fusion for ROS2

This repository contains a Python translation of VINS-Fusion running on ROS2.

The Python ROS2 workspace lives at the repository root and contains four packages:

| ROS2 package | Python project | Description |
| --- | --- | --- |
| `vins_camera_models` | `vins-camera-models` | Camera projection models |
| `vins` | `vins` | Visual-inertial estimator |
| `vins_loop_fusion` | `vins-loop-fusion` | ORB/BFMatcher loop fusion |
| `vins_global_fusion` | `vins-global-fusion` | GPS/global pose fusion |

## Requirements

- ROS2 with `rclpy`, `cv_bridge`, `tf2_ros`, `sensor_msgs`, `nav_msgs`, and `geometry_msgs`
- Python 3.10+
- `uv`

Source ROS2:

```bash
source /opt/ros/<ros2-distro>/setup.bash
```

## Install Python Dependencies

From the repository root:

```bash
uv sync
```

## Build The ROS2 Python Packages

```bash
colcon build --symlink-install
```

Source the overlay:

```bash
source install/setup.bash
```

## Run VINS

Run the estimator directly:

```bash
ros2 run vins vins_node config/euroc/euroc_stereo_imu_config.yaml
```

Or launch it:

```bash
ros2 launch vins vins.launch.py \
  config_path:=config/euroc/euroc_stereo_imu_config.yaml
```

Other useful configs:

```bash
ros2 run vins vins_node config/euroc/euroc_mono_imu_config.yaml
ros2 run vins vins_node config/euroc/euroc_stereo_config.yaml
ros2 run vins vins_node config/realsense_d435i/realsense_stereo_imu_config.yaml
```

## Run Loop Fusion

Start VINS first, then in another terminal:

```bash
cd /media/weiliu/SSD/code/vins_fusion_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run vins_loop_fusion loop_fusion_node config/euroc/euroc_stereo_imu_config.yaml
```

The loop fusion node subscribes to:

- `/vins_estimator/odometry`
- `/cam0/image_raw`

It publishes:

- `/loop_fusion/path`

## Run Global GPS Fusion

Start VINS first, then in another terminal:

```bash
cd /media/weiliu/SSD/code/vins_fusion_ros2
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 run vins_global_fusion global_fusion_node
```

The global fusion node subscribes to:

- `/vins_estimator/odometry`
- `/gps`

It publishes:

- `/global_fusion/odometry`
- `/global_fusion/path`

## Play ROS2 Bags

In a terminal with the ROS environment sourced:

```bash
ros2 bag play /path/to/bag
```

For EuRoC-style data, make sure the bag topics match the config file, for example:

- `/imu0`
- `/cam0/image_raw`
- `/cam1/image_raw`

## Convert ROS1 Bags

If your dataset is still in ROS1 bag format:

```bash
uv pip install rosbags
rosbags-convert /path/to/input.bag --dst /path/to/output_ros2
```

Then play the converted ROS2 bag:

```bash
ros2 bag play /path/to/output_ros2
```

For normal ROS2 usage, prefer the `colcon build` and `ros2 run` workflow above.
