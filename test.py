import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import math
import pandas as pd
import numpy as np
from scipy import ndimage
import json
import gobject

from dat_file import Data


df = pd.DataFrame([[1, 1, 3, 1, 1],
				   [2, 1, 2, 1, 2],
				   [3, 1, 4, 1, 3],
				   [1, 2, 3, 2, 1],
				   [2, 2, 2, 2, 1],
				   [3, 2, 4, 2, 1],
				   [1, 3, 3, 3, 1],
				   [2, 3, 2, 3, 1],
				   [3, 3, 4, 3, 1],])

a = np.array([1,2,3,4])
b = a
b[0] = 234
print a

"""
x = np.array([[0, 1, 2, 3],
			  [0, 1, 2, 3],
			  [0, 1, 2, 3],
			  [0, 1, 2, 3]])

y = np.array([[1, 0, 0, 0],
			  [0, 1, 1, 1],
			  [2, 2, 2, 2],
			  [3, 3, 3, 3]])

z = np.array([[0, 1, 2, 3],
			  [1, 2, 3, 4],
			  [2, 3, 4, 5],
			  [3, 4, 5, 6]])

d = Data(x, y, z)

for x in d.get_sorted():
	print x
	print '\n'
"""