import sys; sys.path.insert(0,"discovery/experiments/scripts")
import R2C_run as R, R2C_solver as S, pandas as pd, numpy as np
df=R.run_split("val",save=False)
d=df[(df.kind==sys.argv[1])&(df.strength>float(sys.argv[2]))]
print(pd.crosstab(d.fault,d.success))
print(d[["fault","family","layers","success","strength","ops"]].head(40).to_string())
