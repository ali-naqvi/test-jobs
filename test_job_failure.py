import ray
import time
ray.init()

for i in range(5):
    print("Count", i)
    time.sleep(1)

raise Exception("Simulate that failure!")
