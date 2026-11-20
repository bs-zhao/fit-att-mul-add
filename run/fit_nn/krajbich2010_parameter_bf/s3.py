"""krajbich2010: S3 parameter inference, frozen source implementation."""
from pipeline import parser,run

if __name__=="__main__":
    a=parser("s3").parse_args()
    run("s3",a)
