create skill_vla directory
git clone robot-skills
git clone spot-sim2real
git checkout joanne/skill-vlm-eval
cd skill_vla
pip install -e
clone spot_ros repo
pip install einops bitsandbytes 
need python 3.10

# Run experiments

# set envirionment variables 
source setup.sh

# launch spot nodes
spot_rl_launch_local

# start eval
cd skill_vla
python experiments/eval_skill_vla.py