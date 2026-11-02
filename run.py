import sys
import os
import importlib.util
import time
import webbrowser
from FreePort8521 import free_port_8521

project_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_dir)

sys.path.insert(0, parent_dir)
spec = importlib.util.spec_from_file_location(
    "recidivism_abm",
    os.path.join(project_dir, "__init__.py"),
    submodule_search_locations=[project_dir]
)

package = importlib.util.module_from_spec(spec)
sys.modules["recidivism_abm"] = package
spec.loader.exec_module(package)

from server import server # assuming your visualization code is in server.py
def launch_mesa_server():
    print("ðŸš€ Launching Mesa server...")
    #os.system("start cmd /c mesa runserver")  # Launches in a new command window
    time.sleep(2)
    server.launch()

if __name__ == "__main__":
    free_port_8521()
    time.sleep(1)
    launch_mesa_server()
