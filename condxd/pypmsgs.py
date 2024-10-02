'''
This code was adopted from PypeIt and the copyright belongs to the PypeIt
developer.
https://github.com/pypeit/PypeIt
'''
import getpass
import inspect
import sys

import astropy
import numpy
import scipy


class PypError(Exception):
    pass

class Messages:
    """
    Create coloured text for messages printed to screen.

    For further details on colours see the following example:
    http://ascii-table.com/ansi-escape-sequences.php

    Parameters
    ----------
    log : str or None
      Name of saved log file (no log will be saved if log=="")
    verbosity : int (0,1,2)
      Level of verbosity:
        0 = No output
        1 = Minimal output
        2 = All output (default)
    colors : bool
      If true, the screen output will have colors, otherwise
      normal screen output will be displayed
    """
    def __init__(self, log=None, verbosity=None, colors=True):

        # Alphabetical list of developers
        developers = ['Riccardo Nanni', 'Jan-Torge Schindler']

        # Initialize other variables
        self._defverb = 1
        if getpass.getuser() in developers:
            self._defverb = 2
        self._verbosity = self._defverb if verbosity is None else verbosity
#        self._last_updated = __last_updated__
        self._version = '0.0a0'

        # TODO: Why are these two necessary?  It would seem better to
        # provide Messages with member functions that can operate on
        # sciexp and jwst_file instead of having them kept within the
        # object itself...
        self.sciexp = None
        self.jwst_file = None

        # Initialize the log
        self._log = None
        self._initialize_log_file(log=log)

        # Use colors?
        self._start = None
        self._end = None
        self._black_CL = None
        self._yellow_CL = None
        self._blue_CL = None
        self._green_CL = None
        self._red_CL = None
        self._white_RD = None
        self._white_GR = None
        self._white_BK = None
        self._white_BL = None
        self._black_YL = None
        self._yellow_BK = None

        self.disablecolors()
        if colors:
            self.enablecolors()

    def _cleancolors(self, msg):
        cols = [self._end, self._start,
                self._black_CL, self._yellow_CL, self._blue_CL, self._green_CL, self._red_CL,
                self._white_RD, self._white_GR, self._white_BK, self._white_BL,
                self._black_YL, self._yellow_BK]
        for i in cols:
            msg = msg.replace(i, '')
        return msg

    def _devmsg(self):
        if self._verbosity == 2:
            info = inspect.getouterframes(inspect.currentframe())[3]
            devmsg = self._start + self._blue_CL + info[1].split('/')[-1] + ' ' + str(info[2]) \
                        + ' ' + info[3] + '()' + self._end + ' - '
        else:
            devmsg = ''
        return devmsg

    def _print(self, premsg, msg, last=True):
        """
        Print to standard error and the log file
        """
        devmsg = self._devmsg()
        _msg = premsg+devmsg+msg
        if self._verbosity != 0:
            print(_msg, file=sys.stderr)
        if self._log:
            clean_msg = self._cleancolors(_msg)
            self._log.write(clean_msg+'\n' if last else clean_msg)

    def _initialize_log_file(self, log=None):
        """
        Expects self._log is already None.
        """
        if log is None:
            return

        # Initialize the log
        self._log = open(log, 'w')

        self._log.write("------------------------------------------------------\n\n")
        self._log.write("This log was generated with version {0:s} of unfold_jwst\n\n".format(
                                                                                    self._version))
        self._log.write("You are using scipy version={:s}\n".format(scipy.__version__))
        self._log.write("You are using numpy version={:s}\n".format(numpy.__version__))
        self._log.write("You are using astropy version={:s}\n\n".format(astropy.__version__))
        self._log.write("------------------------------------------------------\n\n")

    def reset(self, log=None, verbosity=None, colors=True):
        """
        Reinitialize the object.

        Needed so that there can be a default object for all modules,
        but also a dynamically defined log file.
        """
        # Initialize other variables
        self._verbosity = self._defverb if verbosity is None else verbosity
        self.reset_log_file(log)
        self.disablecolors()
        if colors:
            self.enablecolors()

    def reset_log_file(self, log):
        if self._log:
            self._log.close()
            self._log = None
        self._initialize_log_file(log=log)

    def error(self, msg, usage=False):
        """
        Print an error message
        """
        premsg = '\n'+self._start + self._white_RD + '[ERROR]   ::' + self._end + ' '
        self._print(premsg, msg)

        # Close log file
        # TODO: This no longer "closes" the QA plots
        self.close()

        # Print command line usage
        if usage:
            self.usage(None)
        #
        raise PypError(msg)
        sys.exit(1)

    def info(self, msg):
        """
        Print an information message
        """
        premsg = self._start + self._green_CL + '[INFO]    ::' + self._end + ' '
        self._print(premsg, msg)

    def info_update(self, msg, last=False):
        """
        Print an information message that needs to be updated
        """
        premsg = '\r' + self._start + self._green_CL + '[INFO]    ::' + self._end + ' '
        self._print(premsg, msg, last=last)

    def test(self, msg):
        """
        Print a test message
        """
        if self._verbosity == 2:
            premsg = self._start + self._white_BL + '[TEST]    ::' + self._end + ' '
            self._print(premsg, msg)

    def warn(self, msg):
        """
        Print a warning message
        """
        premsg = self._start + self._red_CL + '[WARNING] ::' + self._end + ' '
        self._print(premsg, msg)

    def bug(self, msg):
        """
        Print a bug message
        """
        premsg = self._start + self._white_BK + '[BUG]     ::' + self._end + ' '
        self._print(premsg, msg)

    def work(self, msg):
        """
        Print a work in progress message
        """
        if self._verbosity == 2:
            #premsgp = self._start + self._black_CL + '[WORK IN ]::' + self._end + '\n'
            premsgs = self._start + self._yellow_CL + '[PROGRESS]::' + self._end + ' '
            #self._print(premsgp+premsgs, msg)
            self._print(premsgs, msg)

    def prindent(self, msg):
        """
        Print an indent
        """
        premsg = '             '
        self._print(premsg, msg)

    def input(self):
        """
        Return a text string to be used to display input required from the user
        """
        premsg = self._start + self._blue_CL + '[INPUT]   ::' + self._end + ' '
        return premsg
    
    def close(self):
        # FIXME: According to a comment there is a log to be closed
        pass

    @staticmethod
    def newline():
        """
        Return a text string containing a newline to be used with messages
        """
        return '\n             '

    @staticmethod
    def indent():
        """
        Return a text string containing an indent to be used with messages
        """
        return '             '

    # Set the colors
    def enablecolors(self):
        """
        Enable colored output text
        """

        # Start and end coloured text
        self._start = '\x1B['
        self._end = '\x1B[' + '0m'

        # Clear Backgrounds
        self._black_CL = '1;30m'
        self._yellow_CL = '1;33m'
        self._blue_CL = '1;34m'
        self._green_CL = '1;32m'
        self._red_CL = '1;31m'

        # Coloured Backgrounds
        self._white_RD = '1;37;41m'
        self._white_GR = '1;37;42m'
        self._white_BK = '1;37;40m'
        self._white_BL = '1;37;44m'
        self._black_YL = '1;37;43m'
        self._yellow_BK = '1;33;40m'

    def disablecolors(self):
        """
        Disable colored output text
        """

        # Start and end coloured text
        self._start = ''
        self._end = ''

        # Clear Backgrounds
        self._black_CL = ''
        self._yellow_CL = ''
        self._blue_CL = ''
        self._green_CL = ''
        self._red_CL = ''

        # Coloured Backgrounds
        self._white_RD = ''
        self._white_GR = ''
        self._white_BK = ''
        self._white_BL = ''
        self._black_YL = ''
        self._yellow_BK = ''



