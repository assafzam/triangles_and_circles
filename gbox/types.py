"""
A Module contains commonly used type in our code for cleaner code!
"""

from typing import Union, List, Tuple, Dict

import numpy as np
from PIL import Image

Number = Union[int, float]
NumPair = Tuple[Number, Number]
IntPair = Tuple[int, int]
IntTriple = Tuple[int, int, int]
NumVector = Tuple[Number, ...]
NumList = List[Number]

NumOrNums = Union[Number, NumVector, NumList]
ImgHW = IntPair
ImgHWC = IntTriple
ImgWH = IntPair

PILImg = Image.Image
NumpyImg = np.ndarray

DatasetMappingFile = Dict[str, Dict[str, List[str]]]
MergedDatasetMappingFile = Union[Dict[str, int], Dict[str, str]]
VersionMappingFile = Union[DatasetMappingFile, MergedDatasetMappingFile]
