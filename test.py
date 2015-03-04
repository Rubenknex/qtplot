import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math
import pandas as pd
import numpy as np
from scipy import ndimage
import json
import gobject

from dat_file import Data

df = pd.DataFrame([[1, 0, 1],
				   [1, 0, 2],
				   [1, 0, 3],
				   [2, 0, 4],
				   [2, 0, 5],
				   [2, 0, 6],
				   [3, 0, 7],
				   [3, 0, 8],
				   [3, 0, 9],])

print (df[1] == 0).all()

print df.groupby(0)[1].apply(lambda x: pd.Series(range(len(x.values)), x.index))