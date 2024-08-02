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

if python -c "import site; print(''.join(site.getsitepackages()))"; then
  site_packages_path=$(python -c "import site; print(''.join(site.getsitepackages()))")
else
  echo 'cant locate site-packages location of current python interpreter; try running python -c "import site; print(''.join(site.getsitepackages()))"'
  exit 1
fi

if sudo cp -rf ./rosbridge_library/* $site_packages_path/rosbridge_library/; then 
  echo "Step 1 complete"
else
  exit 1
fi

if sudo cp -rf ./rosbridge_server/* $site_packages_path/rosbridge_server/; then 
  echo "Step 2 complete"
else
  exit 1
fi

if sudo cp -rf ./rosbridge_tcp.launch $CONDA_PREFIX/share/rosbridge_server/launch/; then 
  echo "Step 3 complete"
else
  exit 1
fi

echo "Patch Applied Successfully"