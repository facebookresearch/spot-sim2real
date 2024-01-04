# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any, List, Optional, Tuple


def map_user_input_to_boolean(prompt):
    """
    Maps user input to boolean

    Args:
        prompt (str): Prompt to display to user

    Returns:
        bool: True if user input is y or yes, False if user input is n or no
    """
    while True:
        user_input = input(prompt + "(y/n): ").strip()
        if user_input.lower() in ["y", "yes"]:
            return True
        elif user_input.lower() in ["n", "no"]:
            return False
        else:
            print("Please enter a valid input - y, yes, n or no")


def conditional_print(message: str, verbose: bool = False):
    """
    Print the message if the verbose flag is set to True

    Args:
        message (str): Message to be printed if verbose is True (can also be f-string)
        verbose (bool): Flag to determine whether to print the message or not
    """
    if verbose:
        print(message)
