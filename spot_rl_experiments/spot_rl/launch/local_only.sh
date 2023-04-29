echo "Killing all tmux sessions..."
tmux kill-session -t roscore
tmux kill-session -t img_pub
tmux kill-session -t propio_pub
tmux kill-session -t tts_sub
sleep 1
echo "Starting roscore tmux..."
tmux new -s roscore -d '$CONDA_PREFIX/bin/roscore'
echo "Starting other tmux nodes.."
tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --local'
tmux new -s propio_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --proprioception'
tmux new -s tts_sub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --text-to-speech'
sleep 3
tmux ls
