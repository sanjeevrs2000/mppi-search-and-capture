#!/bin/bash
# Basic entrypoint for ROS / Colcon Docker containers

# Source ROS 2
source /opt/ros/${ROS_DISTRO}/setup.bash
echo "Sourced ROS 2 ${ROS_DISTRO}"

# # Keep GPU memory fragmentation low for PyTorch
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# # Expose PyTorch's bundled CUDA 12 libs so onnxruntime-gpu can find them
# export LD_LIBRARY_PATH=$(python3 -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"):${LD_LIBRARY_PATH}

# Source the base workspace, if built
if [ -f /vrx_ws/install/setup.bash ]
then
  source /vrx_ws/install/setup.bash
  echo "Sourced vrx_ws base workspace"
fi

# Execute the command passed into this entrypoint
exec "$@"
