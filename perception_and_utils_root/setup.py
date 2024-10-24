# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import setuptools

setuptools.setup(
    name="perception_and_utils",
    version="0.0",
    author="Kavit Shah",
    author_email="kavits98@gmail.com",
    description="SIRo friendly perception and utils for Aria+Quest3 and Spot",
    packages=setuptools.find_packages(),
    install_requires=[
        "click",
        "transformers",
    ],
)
