# MPPI-based Informative Trajectory Planning for Search and Capture of Drifting Targets with ASVs
Code accompanying the paper *"MPPI-based Informative Trajectory Planning for Search and Capture of Drifting Targets with ASVs"*.

The system uses an autonomous surface vehicle (WAM-V) to jointly explore an unknown environment and intercept targets that drift freely. A hybrid planner switches between a global exploration mode using the MPPI informative planner and an intercept mode using pure pursuit guidance.

Simulations are built on the [VRX](https://github.com/osrf/vrx) simulator (ROS 2 Humble, Gazebo Garden, Ubuntu 22.04).

## Packages

| Package | Description |
|---|---|
| `informative_planning` | MPPI informative planner, dynamic occupancy grid map, hybrid guidance node |
| `usv_navigation` | Stereo-vision object detector and PD velocity controller |
<!-- | `pos_publisher` | Gazebo–ROS bridge for ground-truth object positions (C++) |
| `vrx_custom_msgs` | Custom ROS 2 service definitions |
| `vrx_gz` / `vrx_ros` / `vrx_urdf` | VRX simulation environment (modified from upstream) | -->

## Setup

```bash
# Build the Docker image
docker build -f Dockerfile -t user/mppi_search_capture:1.0 .

# Start a container
chmod +x run_docker.sh
./run_docker.sh
```

Inside the container:

```bash
cd /vrx_ws
colcon build --merge-install
source install/setup.bash
```

## Running experiments

Launch a single trial (with logging):

```bash
ros2 launch informative_planning hybrid_plan_log.launch.py world:=exp1_speed12 headless:=True
# set headless:=False to view the Gazebo GUI
```

For headless batch runs, edit `world` and `N_TRIALS` in `run_sims.sh`, then:

```bash
./run_sims.sh
```

Available worlds: `exp1_speed12`, `exp2_speed9`, `exp3_speed8` — corresponding to the low, medium, and high target density scenarios in the paper.

To visualise with RViz in a separate terminal:

```bash
ros2 launch informative_planning vis_rviz.launch.py
```

All parameters (planner selection, cost weights, sensor range, mission time, mode-switching threshold, etc.) can be changed in `src/informative_planning/informative_planning/params.py`. MPPI sampling hyperparameters (number of samples, horizon, noise covariance, action limits) are in `src/informative_planning/informative_planning/mppi_config.yaml`.

## License

This repository is released under the [MIT License](LICENSE), except for third-party code that keeps its original license:

- `vrx_gz`, `vrx_ros`, `vrx_urdf`: modified from [VRX](https://github.com/osrf/vrx), Apache License 2.0. The launch file `hybrid_plan_log.launch.py` is adapted from VRX and keeps its Apache 2.0 header.
- `informative_planning/mppi_torch/mppi_utils.py`: MIT License, Copyright (c) 2020-2021 NVIDIA CORPORATION.
