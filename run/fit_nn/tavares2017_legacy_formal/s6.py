"""Frozen legacy Tavares2017 G2 S6 entry point."""
from core import parser, run

if __name__ == "__main__":
    args=parser("s6").parse_args()
    run("s6",args)
