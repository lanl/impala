""" timeout context handler -- calls a timeout error if takes too long between
__enter__ and __exit__.

credit to Thomas Alhe:
https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
"""
import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

# EOF
