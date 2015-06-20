# qtplot
Plotting program inspired by Spyview.

Reads .dat files created by qtlab.

## Running qtplot
The python executable must be in the PATH environment variable.

qtplot can be run on windows using the `qtplot.bat` file.

To open a `.dat` file by clicking, select `qtplot.bat` in its "Open with:" menu.

## Automation using standalone `data.py`

By including the `data.py` file next to a personal Python script, all the core data operations and some plotting capabilities can be used. This can be useful for automating certain calculations on multiple datasets.

```
import matplotlib.pyplot as plt
from data import DatFile

df = DatFile('some_qtlab_data.dat')
data = df.get_data('X measure', 'Y measure', 'Z measure', 'X bias', 'Y bias')

data.crop(right=-10)
data.lowpass(x_width=3, y_height=0, method='gaussian')
data.xderiv()

plt.pcolormesh(*data.get_pcolor(), cmap='seismic')
```

## Dependencies
#### Recommended
Install a scientific Python distribution such as Anaconda (https://store.continuum.io/cshop/anaconda/). For now only Python 2.7 is tested.

The only additional packages needed this way are vispy and pyopengl which can be installed with `pip install vispy pyopengl`.

##### Python 2.7
Not tested on earlier versions but might work.

##### NumPy >= 1.9.0
Download the right .whl for your Python version and 32/64 bits:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy

Install with `pip install <.whl file>`

##### SciPy >= 0.15.0
Download the .exe installer at 
http://sourceforge.net/projects/scipy/

OR, Download the right .whl for your Python version and 32/64 bits:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#scipy

Install with `pip install <.whl file>`

##### PyQt4 >= 4.11.0
Download the right .whl for your Python version and 32/64 bits:
http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4

Install with `pip install <.whl file>`

##### matplotlib / pandas / vispy / pyopengl
matplotlib >= 1.4.2

pandas >= 0.16.0

vispy >= 0.4.0

pyopengl >= 3.1.0

Install with `pip install matplotlib pandas vispy pyopengl`
