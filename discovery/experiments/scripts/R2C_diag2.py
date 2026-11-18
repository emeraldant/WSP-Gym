import sys; sys.path.insert(0,"discovery/experiments/scripts")
import R2C_run as R, R2C_solver as S, pandas as pd, numpy as np
df=R.run_split("val",save=False)
df["clean"]=df.fault=="clean"
df["sbin"]=pd.cut(df.strength,[0,.2,.25,.3,.4,.5,.7,.9,1.01,3])
g=df.groupby(["kind","sbin"],observed=True).agg(n=("success","size"),succ=("success","sum"),clean=("clean","sum"))
print(g.to_string())
print("no-edit baseline overall", df.clean.mean())
