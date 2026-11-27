"""tavares2017 Final2 S1 stage."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from final2_launcher import main
if __name__=="__main__":
    main(["s1","--dataset","tavares2017",*sys.argv[1:]])
