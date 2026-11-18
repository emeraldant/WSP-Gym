# Runs the FROZEN spec-free solver ONCE on the six test splits (val already tuned).
import sys; sys.path.insert(0,"discovery/experiments/scripts")
import R2C_run as R, hashlib
assert hashlib.md5(open("discovery/experiments/outputs/R2C/R2C_solver_FROZEN.py","rb").read()).hexdigest()=="3855718bae53bdfec17ea2049e169051"
assert open("discovery/experiments/outputs/R2C/R2C_solver_FROZEN.py").read()==open("discovery/experiments/scripts/R2C_solver.py").read()
R.run_split("val")
for sp in ["test_id","test_larger","test_families","test_faults","test_trained","test_external"]:
    df=R.run_split(sp); print(sp,"faulty %.3f clean %.3f overall %.3f errs %d"%(*R.summ(df),df.err.sum()),flush=True)
