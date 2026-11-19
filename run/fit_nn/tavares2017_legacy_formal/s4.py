"""Frozen legacy Tavares2017 G2 S4 entry point."""
from core import parser, run

if __name__ == "__main__":
    args=parser("s4").parse_args()
    run("s4",args)
