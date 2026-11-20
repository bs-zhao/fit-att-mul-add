"""krajbich2010 Final1 S5 stage."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final1_launcher import main
if __name__=="__main__":
    main(["s5","--dataset","krajbich2010",*sys.argv[1:]])
