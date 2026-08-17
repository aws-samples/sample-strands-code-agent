"""
Register a botocore hook that appends the AWS Solutions ID of this library to
the User-Agent tags of the botocore calls.
"""
import botocore.session

SOLUTION_UA = "AWSSOLUTION/SO0353/v0.4.0"


_orig_init = botocore.session.Session.__init__


def _init(self, *args, **kwargs):
    _orig_init(self, *args, **kwargs)
    extra = self.user_agent_extra or ""
    if SOLUTION_UA and SOLUTION_UA not in extra.split():
        self.user_agent_extra = f"{extra} {SOLUTION_UA}".strip()


botocore.session.Session.__init__ = _init
