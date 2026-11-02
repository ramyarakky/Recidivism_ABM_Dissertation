import psutil

def free_port_8521():
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == 8521 and conn.status == psutil.CONN_LISTEN:
            pid = conn.pid
            if pid:
                try:
                    p = psutil.Process(pid)
                    p.terminate()
                    print(f"✅ Terminated process on port 8521 (PID: {pid})")
                except Exception as e:
                    print(f"⚠️ Failed to terminate PID {pid}: {e}")
            else:
                print("⚠️ No PID found for port 8521")
            break
    else:
        print("✅ Port 8521 is already free.")
