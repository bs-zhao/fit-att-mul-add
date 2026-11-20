"""tavares2017 Final1 S6 stage."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final1_launcher import main
if __name__=="__main__":
    main(["s6","--dataset","tavares2017",*sys.argv[1:]])
