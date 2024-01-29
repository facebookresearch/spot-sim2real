# Atomic Skills

## Atomic Skills is a skill which requires its own policy and environment and does not depend on other atomic or composite skills

## Follow the following class design to implement any atomic skill of your own

```bash

Process to add a new atomic skill:

class NewAtomicSkill(Skill):
    """
    <Add docstring explaining the skill>
    <Add `Expected goal_dict input` in the docstring>
    <Add `Args` in the docstring>
    <Add `How to use` in the docstring>
    <Add `Example` in the docstring>
    """

    def __init__(self, spot: Spot, config=None) -> None:
        # <Construct config if not provided>
        super().__init__(spot, config)

        # <Update this depending on the NewPolicy and NewEnv design>
        # Setup
        self.policy = NewPolicy(
            self.config.WEIGHTS.NEW, device=self.config.DEVICE, config=self.config
        )
        self.policy.reset()

        self.env = NewEnv(self.config, self.spot)

    def sanity_check(self, goal_dict: Dict[str, Any]):
        """
        Refer to class Skill for documentation
        @ Note: Defining this method is not explicitly required and one may skip to do so.
                But its recommended to verify input being used.
        """
        <Update sanity check based on goal keys. Make sure all required keys are verified>
        new_target = goal_dict.get(<goal_name>, None) # type: <XYZ>
        if new_target is None:
            raise KeyError(
                "Error in NewAtomicSkill.sanity_check(): new_target key not found in goal_dict"
            )

    def reset_skill(self, goal_dict: Dict[str, Any]) -> Any:
        """Refer to class Skill for documentation"""
        try:
            self.sanity_check(goal_dict)
        except Exception as e:
            raise e

        <Print what this skill will do in this execution loop>
        conditional_print(
            message=f"New target at {new_target}.",
            verbose=self.verbose,
        )

        # Reset the env and policy
        observations = self.env.reset(new_target)
        self.policy.reset()

        # Logging and Debug
        self.env.say(f"NewAtomicSkill at {new_target}")

        # Reset logged data at init
        self.reset_logger()

        return observations

    def execute(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:  # noqa
        <Use execute method from base class unless there are different steps that need to be performed (like place.execute())>

    def update_and_check_status(self, goal_dict: Dict[str, Any]) -> Tuple[bool, str]:
        """Refer to class Skill for documentation"""
        # <Update the logged data at the end of the skill execution loop (as defined in class Skill) and return feedback status>

    def split_action(self, action: np.ndarray) -> Dict[str, Any]:
    """Refer to class Skill for documentation"""
        # <Split action output from policy into an action dictionary (based on NewPolicy output)>
        # Return the action dictionary

```


## RULES:
1. All atomic skills should inherit from Skill class
2. All policy based atomic skills should have their own policy and environment and should not use other atomic skill objects for their execution
3. All atomic skills should ONLY process 1 goal at a time
4. All policy based atomic skills should use the execute_rl_loop() method for the execution while loop


# Composite Skills

## Composite skill is a skill which DOES NOT have its own policy and environment and orchestrates its actions based on sequential composition of other atomic / composite skills