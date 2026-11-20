"""food_equal: S2 parameter inference, frozen source implementation."""
from pipeline import parser,run

if __name__=="__main__":
    a=parser("s2").parse_args()
    run("s2",a)
