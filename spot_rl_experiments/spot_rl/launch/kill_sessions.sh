echo "Killing all tmux sessions..."
tmux kill-session -t roscore
tmux kill-session -t headless_estop
tmux kill-session -t img_pub
tmux kill-session -t propio_pub
tmux kill-session -t tts_sub
tmux kill-session -t remote_spot_listener
tmux kill-session -t spotWorld_static_tf2_pub
echo "Here are your remaining tmux sessions:"
tmux ls
