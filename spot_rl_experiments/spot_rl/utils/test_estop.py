import threading
import time

import rospy


def read_emergency_stop(force_stop=True):
    while True:
        estop = rospy.get_param("estop", False)
        if estop:
            break


thread = threading.Thread(target=read_emergency_stop, args=(True,))
thread.start()
print("Spun off thread")
