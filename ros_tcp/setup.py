from setuptools import find_packages, setup

setup(
    name="ros_communication_client",
    version="0.1.0",
    packages=find_packages(include=['ros_communication_client', 'ros_communication_client.*']),
    install_requires=["scipy", "pymongo"],
    description="ROS Communication Client Module",
    author="Tushar Sangam",
    author_email="sangamtushar@meta.com",
    url="https://github.com/facebookresearch/spot-sim2real",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
