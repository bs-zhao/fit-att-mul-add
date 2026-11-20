"""food_equal Final1 S3 stage."""
import sys
from run import main
if __name__=="__main__":
    main(["s3",*sys.argv[1:]])
