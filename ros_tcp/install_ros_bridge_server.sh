#!/bin/bash
# Check if $CONDA_PREFIX contains the substring 'spot_ros'
# Install a package using mamba
package_name="ros-noetic-rosbridge-server"

echo "Installing $package_name -c robostack -c conda-forge -c robostack-experimental -y"
if mamba install -y $package_name; then
  echo "$package_name installed successfully."
else
  echo "Error: Failed to install $package_name."
  exit 1
fi
pip install pymongo
echo "Applying patch to "$CONDA_PREFIX
sudo cp -rf ./rosbridge_library/* $CONDA_PREFIX/lib/python3.8/site-packages/rosbridge_library/
sudo cp -rf ./rosbridge_server/* $CONDA_PREFIX/lib/python3.8/site-packages/rosbridge_server/
sudo cp -rf ./rosbridge_tcp.launch $CONDA_PREFIX/share/rosbridge_server/launch/
echo "Patch Applied Successfully"