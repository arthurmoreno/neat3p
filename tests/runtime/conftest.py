import os
import sys

# Test helpers (make_fixtures, _parity) live next to the tests, imported by basename.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
