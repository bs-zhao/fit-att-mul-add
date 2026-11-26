"""S3-fitted RACE-Add -> pretrained DDM4, tavares2017 Final1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final1_s3_fitted_cross import run

if __name__ == "__main__":
    run(["--dataset", "tavares2017", *sys.argv[1:]])
