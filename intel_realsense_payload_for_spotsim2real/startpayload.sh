#!/bin/bash
export SPOT_IP=192.168.50.3
export SPOT_ADMIN_PW=i4fhwamvx5rf
export PAYLOAD_GUID=78b076a2-b4ba-491d-a099-738928c4410c
export PAYLOAD_SECRET=spot-sim2realteamsiro
export PAYLOAD_IP=192.168.50.5
export PAYLOAD_PORT=21000
export PAYLOAD_NAME=IntelRealSenseAuxillaryImageServicePayload

cd /home/spot/Desktop/intel_realsense_payload_for_spotsim2real
echo "Starting the execution"
python3 intelrealsense_image_service.py --guid $PAYLOAD_GUID \
--secret $PAYLOAD_SECRET --host-ip $PAYLOAD_IP \
--port $PAYLOAD_PORT $SPOT_IP --jpeg-quality 75 --png-quality 5 --res-width 640 --res-height 480 --filter-depth #\
#--socket-verification #--show-debug-info
