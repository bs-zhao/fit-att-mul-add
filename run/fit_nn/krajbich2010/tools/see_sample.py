import pickle
from pathlib import Path
p = Path(__file__).resolve().parents[4] / 'outputs/krajbich2010/s3_fe1/aDDM_t/gen1_batch0.pkl'
with p.open('rb') as f:
    x = pickle.load(f)
print(x[0] if x else 'No subjects stored')
