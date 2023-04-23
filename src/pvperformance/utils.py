from pathlib import Path
from enum import Enum

vxMatchWeight = 0.7
cutMinTruthRecoRadialDiff = 0.1

class VertexMatchType(Enum):
    """ Copy the definitions here
    https://acode-browser1.usatlas.bnl.gov/lxr/source/athena/InnerDetector/InDetValidation/InDetTruthVertexValidation/InDetTruthVertexValidation/InDetVertexTruthMatchUtils.h?v=21.9#0020
    """
    MATCHED = 0
    MERGED = 1
    SPLIT = 2
    FAKE = 3
    DUMMY = 4
    NTYPES = 5

class HardScatterType(Enum):
    """ Copy the definitions here
    https://acode-browser1.usatlas.bnl.gov/lxr/source/athena/InnerDetector/InDetValidation/InDetTruthVertexValidation/InDetTruthVertexValidation/InDetVertexTruthMatchUtils.h?v=21.9#0030
    """
    CLEAN = 0
    LOWPU = 1
    HIGHPU = 2 
    HSSPLIT = 3
    NONE = 4
    NHSTYPES = 5

def check_outputpath(output_path):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path 

