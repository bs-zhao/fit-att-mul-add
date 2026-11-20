"""krajbich2010: S1 parameter inference, frozen source implementation."""
from pipeline import parser,run

if __name__=="__main__":
    a=parser("s1").parse_args()
    run("s1",a)
