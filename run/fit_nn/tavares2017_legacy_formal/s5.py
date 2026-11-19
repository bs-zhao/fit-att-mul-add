"""Frozen legacy Tavares2017 G2 S5 entry point."""
from core import parser, run

if __name__ == "__main__":
    args=parser("s5").parse_args()
    run("s5",args)
