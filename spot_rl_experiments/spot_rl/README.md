# Skill Development
Spot-Sim2Real is a modular library for the development of Spot for embodied AI tasks. One of the key features is to add a neural network skill. We illustrate the design of skill execution and the steps to add new skills.

# Skill Codebase Design
1. ```spot_rl_experiments/spot_rl/real_policy.py```:  RealPolicy is a class that is used to define the observation and action spaces of skills and load the weights of the neural network skill via a torchscript approach. When you train a new skill with a new weight, you add a new skill here for loading the weight. For how to convert the skill into torchscript format, please follow the [instruction](spot_rl_experiments/utils/README.md).
2. ```spot_rl_experiments/spot_rl/skills/atomic_skills.py```: Atomic Skills is a skill which requires its own policy and environment and does not depend on other atomic or composite skills. Please follow the [instruction](spot_rl_experiments/spot_rl/skills/README.md) to add a new skill.
3. ```spot_rl_experiments/spot_rl/envs/skill_manager.py```: This class exposes the high-level API to users and loads the skills defined in ```atomic_skills.py``.
4. Define your own ```skill_env.py``` under```spot_rl_experiments/spot_rl/envs/```: This class defines how you want to organize the observation and action spaces from the sensor reading.
The following figure is a high-level overview.
![](spot-sim2real/doc/skill_design.png)<br>
