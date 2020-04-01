import os
import ctypes

try:
    from knn_python import *
except:
    script_dir = os.path.dirname(os.path.dirname(__file__))
    os.environ['LD_LIBRARY_PATH'] += (script_dir)
    ctypes.cdll.LoadLibrary(os.path.join(script_dir, "libtensorflow.so"))
    ctypes.cdll.LoadLibrary(os.path.join(script_dir, "libtensorflow_framework.so"))
    from knn_python import *

