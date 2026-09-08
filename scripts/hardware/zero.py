import time

from piper_sdk import C_PiperInterface_V2

# Gripper angle in 0.001 mm units. 70 mm is the Piper gripper's full open.
OPEN_POSITION = 70_000


def main():
    arms = {
        "left": C_PiperInterface_V2("can_arm_left"),
        "right": C_PiperInterface_V2("can_arm_right"),
    }

    for arm in arms.values():
        arm.ConnectPort()
    time.sleep(0.1)

    for arm in arms.values():
        while not arm.EnablePiper():
            time.sleep(0.01)

    for arm in arms.values():
        arm.ModeCtrl(0x01, 0x01, 30, 0x00)

    for _ in range(5):
        for arm in arms.values():
            arm.JointCtrl(0, 0, 0, 0, 0, 0)
            arm.GripperCtrl(OPEN_POSITION, 1000, 0x01, 0)
        time.sleep(0.5)

    arms["right"].GripperCtrl(0, 1000, 0x01, 0)


if __name__ == "__main__":
    main()
