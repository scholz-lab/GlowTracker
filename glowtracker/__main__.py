# To support deployment as a module, we need to 
#   1. Change the CWD to current deployment location
import os
curr_file_abs_path = os.path.abspath(__file__)
curr_file_dir = os.path.dirname(curr_file_abs_path)
os.chdir(curr_file_dir)

#   2. Add the CWD to PYTHONPATH
import sys
sys.path.insert(0, curr_file_dir)

# Disable kivy console log
os.environ["KIVY_NO_CONSOLELOG"] = "1"

# Start application
from GlowTracker import main
main()