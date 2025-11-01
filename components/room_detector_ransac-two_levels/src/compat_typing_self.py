# compat_typing_self.py
# Must be imported before torch or anything that imports torch._dynamo

import sys
import typing

# Only shim on Python < 3.11
if sys.version_info < (3, 11):
    try:
        from typing_extensions import Self as _Self  # pip install typing_extensions
        # Make typing.Self point to the compatible version
        setattr(typing, "Self", _Self)
    except Exception:
        # If typing_extensions is missing, fail silently (you can also log a warning).
        pass
