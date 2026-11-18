import sys; sys.path.insert(0,"discovery/experiments/scripts")
import R2C_run as R, R2C_solver as S, pandas as pd
df=R.run_split("val",save=False)
df["isclean"]=df.fault=="clean"
print(df.groupby(["kind","isclean"]).success.agg(["mean","count"]).round(2))
print(df[(df.kind=="cl_mixed")][["fault","success","strength","family"]].head(30).to_string())
