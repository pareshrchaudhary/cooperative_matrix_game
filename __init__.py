import os
import sys
package_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(package_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
