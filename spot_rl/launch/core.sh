echo "Killing all tmux sessions..."
tmux kill-session -t roscore
tmux kill-session -t headless_estop
tmux kill-session -t img_pub
tmux kill-session -t propio_pub
tmux kill-session -t tts_sub
tmux kill-session -t remote_spot_listener
sleep 2
echo "Starting roscore tmux..."
tmux new -s roscore -d '$CONDA_PREFIX/bin/roscore'
sleep 1
echo "Starting other tmux nodes"
tmux new -s headless_estop -d '$CONDA_PREFIX/bin/python -m spot_wrapper.headless_estop'
tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --core'
tmux new -s propio_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --proprioception'
tmux new -s tts_sub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.helper_nodes --text-to-speech'
tmux new -s remote_spot_listener -d 'while true; do $CONDA_PREFIX/bin/python -m spot_rl.utils.remote_spot_listener ; done'
sleep 3
tmux ls
