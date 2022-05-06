import ray
import time
ray.init()

for i in range(600):
    print("Count", i)
    time.sleep(1)

raise Exception("Simulate that failure!")
    

