# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import subprocess


def update_iptables(protocol:str="tcp", direction:str="INPUT") -> None:
    """
    Update firewall to permit incoming tcp / udp connections for DDS
    """
    update_iptables_cmd = [
        "sudo",
        "iptables",
        "-A",
        direction,
        "-p",
        protocol,
        "-m",
        protocol,
        "--dport",
        "7000:8000",
        "-j",
        "ACCEPT",
    ]
    print("Running the following command to update iptables:")
    print(update_iptables_cmd)
    subprocess.run(update_iptables_cmd)

def update_iptables_aria():
    update_iptables(protocol="udp", direction="INPUT")

def update_iptables_quest3():
    update_iptables(protocol="tcp", direction="INPUT")
    update_iptables(protocol="tcp", direction="OUTPUT")
