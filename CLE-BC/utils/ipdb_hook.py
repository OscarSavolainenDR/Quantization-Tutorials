import traceback, ipdb
import sys


def ipdb_sys_excepthook():
    """
    When called this function will set up the system exception hook.
    This hook throws one into an ipdb breakpoint if and where a system
    exception occurs in one's run.

    E.g.
    >>> ipdb_sys_excepthook()
    """

    def info(type, value, tb):
        """
        System excepthook that includes an ipdb breakpoint.
        """
        if hasattr(sys, "ps1") or not sys.stderr.isatty():
            # we are in interactive mode or we don't have a tty-like
            # device, so we call the default hook
            sys.__excepthook__(type, value, tb)
        else:
            # we are NOT in interactive mode, print the exception...
            traceback.print_exception(type, value, tb)
            print
            # ...then start the debugger in post-mortem mode.
            # pdb.pm() # deprecated
            ipdb.post_mortem(tb)  # more "modern"

    sys.excepthook = info
