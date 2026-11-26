"""food_equal Final2 S6 stage."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final2_launcher import main
if __name__=="__main__":
    main(["s6","--dataset","food_equal",*sys.argv[1:]])
