"""S3-fitted RACE-Add -> pretrained DDM4, food_equal Final1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from final1_s3_fitted_cross import run

if __name__ == "__main__":
    run(["--dataset", "food_equal", *sys.argv[1:]])
