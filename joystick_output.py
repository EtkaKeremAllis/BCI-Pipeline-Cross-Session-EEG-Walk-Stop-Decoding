"""
joystick_output.py
==================
prediction -> state machine -> joystick output

The current layer is SAFE: it does not connect to a real joystick driver
(vJoy/ViGEm), it just prints the command to the terminal. If real hardware
is added later, only the two branches inside JoystickOutput.send() need to
be filled in; prediction_to_command() and the outward-facing API stay the same.
"""


def prediction_to_command(prediction: int, confidence: float, threshold: float = 0.6) -> str:
    """
    Converts a raw (prediction, confidence) pair into a WALK / STOP / IDLE command.

    - if confidence is below the threshold -> IDLE (don't trust an unstable/noisy prediction)
    - prediction == 1            -> WALK
    - prediction == 0            -> STOP
    """
    if confidence < threshold:
        return "IDLE"

    if prediction == 1:
        return "WALK"

    return "STOP"


class JoystickOutput:
    """
    For now, a safe output layer.
    Prints the command to the terminal until real vJoy/ViGEm support is added.

    State changes are tracked via last_command: when the same command arrives
    repeatedly (e.g. every window predicting WALK during a WALK period), the
    print/axis update is not repeated over and over - an action is only
    triggered WHEN THE COMMAND CHANGES (STOP->WALK or WALK->STOP).
    If a continuous joystick axis feed is needed (e.g. keeping the forward
    axis live on every frame), this behavior can be adjusted inside send();
    for now the design works on a "trigger on state change" basis.
    """

    def __init__(self):
        self.last_command = None

    def send(self, command: str):
        if command == self.last_command:
            return

        self.last_command = command

        if command == "WALK":
            print("[JOYSTICK] FORWARD")
            # Real vJoy forward axis will go here later

        elif command == "STOP":
            print("[JOYSTICK] STOP")
            # Real joystick neutral will go here later

        else:
            print("[JOYSTICK] IDLE")


# ============================================================================
# Simple usage (single window)
# ============================================================================
if __name__ == "__main__":
    joystick = JoystickOutput()

    # Example: assume pipe.predict([current_window]) -> (pred, conf)
    # pred, conf = pipe.predict([current_window])
    # command = prediction_to_command(int(pred[0]), float(conf[0]), threshold=0.6)
    # joystick.send(command)

    # Manual test:
    for pred, conf in [(1, 0.9), (1, 0.85), (0, 0.7), (0, 0.4)]:
        command = prediction_to_command(pred, conf, threshold=0.6)
        joystick.send(command)
