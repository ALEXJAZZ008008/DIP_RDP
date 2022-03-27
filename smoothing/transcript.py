# Copyright University College London 2022
# Author: Alexander Whitehead, Institute of Nuclear Medicine, UCL
# For internal research only.


"""
https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
Transcript - direct print output to a file, in addition to terminal.

Usage:
    import transcript
    transcript.start("logfile.log")
    print("inside file")
    transcript.stop()
    print("outside file")
"""


import sys


class Transcript(object):
    def __init__(self, filename=None):
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        if filename is not None:
            self.logfile = open(filename, "a")
        else:
            self.logfile = None

    def write(self, message):
        self.stdout.write(message)

        if self.logfile is not None:
            self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.

        pass


def transcript_start(filename):
    """Start transcript, appending print output to given filename"""

    logfile = Transcript(filename)

    sys.stdout = logfile
    sys.stderr = logfile

    return logfile


def transcript_stop(logfile=Transcript()):
    """Stop transcript and return print functionality to normal"""

    try:
        logfile.logfile.close()

        sys.stdout = logfile.stdout
        sys.stderr = logfile.stderr
    except:
        pass

    return True