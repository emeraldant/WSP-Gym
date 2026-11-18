import time
from weightgym.env import GymConfig, make_episode
for src in ["trained","compiled"]:
    t=time.time(); n=0
    for i in range(12):
        cfg=GymConfig(size=i%2, source=src)
        make_episode(30_000_000+i, cfg); n+=1
    print(src,(time.time()-t)/n,"s/ep",flush=True)
