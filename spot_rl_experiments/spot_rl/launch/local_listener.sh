echo "Killing img_pub sessions..."
tmux kill-session -t img_pub
sleep 1
echo "Starting img_pub session"
tmux new -s img_pub -d '$CONDA_PREFIX/bin/python -m spot_rl.utils.img_publishers --listen'
sleep 3
tmux ls
