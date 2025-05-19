"""
Attributor module for computing influence and other attributions.
"""

from .attributor import IFAttributor
from .factory import create_attributor

# Export public API
__all__ = ["IFAttributor", "create_attributor"]