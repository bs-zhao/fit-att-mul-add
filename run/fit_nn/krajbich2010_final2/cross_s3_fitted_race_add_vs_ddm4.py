"""krajbich2010 Final2 S3-fitted RACE-Add -> four DDM."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final2_s3_fitted_cross import run

if __name__ == "__main__":
    run(["--dataset", "krajbich2010", *sys.argv[1:]])
