from pathlib import Path as _Path
import os as _os
_os.chdir(_Path(__file__).resolve().parent)
import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).with_name("s5_fe_real_data_maxT.py")),
    run_name="__main__",
)
