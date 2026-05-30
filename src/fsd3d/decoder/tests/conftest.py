"""pytest configuration for decoder tests.

Adds the project src/ directory to sys.path so the fsd3d package
can be imported when running tests from the decoder/ directory.
"""

import sys
import os

# Add project src/ to path (3 levels up from this conftest.py)
_project_src = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "src"))
if _project_src not in sys.path:
    sys.path.insert(0, _project_src)
