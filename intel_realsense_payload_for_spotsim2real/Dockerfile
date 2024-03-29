# Use the specified base image
FROM dustynv/realsense:r36.2.0

# Clone the specific branch of the repository
COPY ./ /spot-sim2real/intel_realsense_payload_for_spotsim2real

# Change the working directory to the repository folder
WORKDIR /spot-sim2real/intel_realsense_payload_for_spotsim2real

# Install Python dependencies from the requirements.txt file
RUN pip3 install -r requirements.txt || true

# Make sure the start script is executable
RUN chmod +x startpayload.sh

# Specify the command to run the start script
CMD ["./startpayload.sh"]
#to run docker without sudo usermod -a -G docker spot
#sudo docker build -t intelrealsensepayload .
#sudo docker run -it --rm --privileged --network host --device /dev:/dev intelrealsensepayload:latest