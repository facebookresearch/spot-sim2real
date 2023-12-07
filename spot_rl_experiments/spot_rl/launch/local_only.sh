echo "Killing all tmux sessions..."
tmux kill-session -t roscore
tmux kill-session -t img_pub
tmux kill-session -t propio_pub
tmux kill-session -t tts_sub
tmux kill-session -t spotWorld_static_tf2_pub
# tmux kill-session -t rviz
sleep 1
echo "Starting roscore tmux..."
tmux new -s roscore -d '$CONDA_PREFIX/bin/roscore'
echo "Starting other tmux nodes.."
tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --local'
tmux new -s propio_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --proprioception'
tmux new -s tts_sub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --text-to-speech'
tmux new -s spotWorld_static_tf2_pub -d 'rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 /map /spotWorld'
# tmux new -s rviz -d 'rosrun rviz rviz -d $CONDA_PREFIX/envs/spot_rl/lib/python3.6/site-packages/spot_rl/utils/spot_world.rviz'
sleep 3
tmux ls

# This for running mask rcnn in img_publishers, which needs input images to be in grayscale
#tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --local --bounding_box_detector mrcnn'
