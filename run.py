import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from server import server # assuming your visualization code is in server.py
import time
import webbrowser
from FreePort8521 import free_port_8521


def launch_mesa_server():
    print("🚀 Launching Mesa server...")
    #os.system("start cmd /c mesa runserver")  # Launches in a new command window
    time.sleep(2)
    server.launch()

if __name__ == "__main__":
    free_port_8521()
    time.sleep(1)
    launch_mesa_server()
