echo "Killing all tmux sessions..."
tmux kill-session -t roscore
tmux kill-session -t img_pub
tmux kill-session -t propio_pub
tmux kill-session -t tts_sub
tmux kill-session -t spotWorld_static_tf2_pub
tmux kill-session -t segmentation_service
tmux kill-session -t pose_estimation_service
tmux kill-session -t ros_bridge_server
tmux kill-session -t spotWorld_static_tf2_pub_to_spot_world
tmux kill-session -t spotWorld_static_tf2_pub_to_spot_world_to_marker
tmux kill-session -t HAR

sleep 1
echo "Starting roscore tmux..."
tmux new -s roscore -d '$CONDA_PREFIX/bin/roscore'
echo "Starting other tmux nodes.."
tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --local'
tmux new -s propio_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --proprioception'
tmux new -s tts_sub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --text-to-speech'
tmux new -s spotWorld_static_tf2_pub_to_spot_world -d 'rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 /map /spotWorld'
tmux new -s spotWorld_static_tf2_pub_to_spot_world_to_marker -d 'rosrun tf2_ros static_transform_publisher -0.67314154  0.03823617  0.01069196 0.5016031  -0.49577103  0.50112072 -0.50148124 /spotWorld /marker'
tmux new -s pose_estimation_service -d 'cd third_party/FoundationPoseForSpotSim2Real/ && sh run_pose_estimation_service.sh'
sleep 3
tmux new -s ros_bridge_server -d 'roslaunch rosbridge_server rosbridge_tcp.launch'
sleep 3
tmux new -s HAR -d 'bash scripts/run_hand_object_detector.sh'
tmux ls

# This for running mask rcnn in img_publishers, which needs input images to be in grayscale
#tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --local --bounding_box_detector mrcnn'
# This is for the sementation service
#tmux new -s segmentation_service -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.segmentation_service'