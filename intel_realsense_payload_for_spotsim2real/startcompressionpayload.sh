#!/bin/bash
export PYTHONPATH=/home/spot/.local/lib/python3.10/site-packages
export SPOT_IP=192.168.50.3
export SPOT_ADMIN_PW=i4fhwamvx5rf
export PAYLOAD_IP=192.168.50.5
export FAST_DEPTH_PORT=21998


cd /home/spot/Desktop/intel_realsense_payload_for_spotsim2real
echo "Starting the execution"
python3 depth_compressor_payload.py
