"""krajbich2010 Final2 S2 stage."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final2_launcher import main
if __name__=="__main__":
    main(["s2","--dataset","krajbich2010",*sys.argv[1:]])
