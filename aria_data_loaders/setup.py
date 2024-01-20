# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import setuptools

setuptools.setup(
    name="aria_data_utils",
    version="0.0",
    author="Priyam Parashar",
    author_email="priyam8parashar@gmail.com",
    description="SIRo friendly data-loaders for Aria data \
    based on Project Aria data utilities",
    packages=setuptools.find_packages(),
    install_requires=[
        "click",
        "transformers",
        "projectaria-tools",
        "projectaria_client_sdk",
    ],
)
