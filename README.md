![alt tag](screenshot.png)

## Installing

The dependencies are:
- Python 2.7 or Python 3 up to 3.4 (vispy doesn't work on 3.5 yet)
- NumPy
- SciPy
- matplotlib
- pandas
- vispy
- pyopengl

It is recommended to install a scientific Python distribution such as Anaconda (https://store.continuum.io/cshop/anaconda/) or Canopy (https://store.enthought.com/downloads) because most scientific libraries as NumPy and SciPy depend on compiled libraries. 

Install the packages that are not included by default:

`pip install vispy pyopengl` or (Anaconda) `conda install vispy pyopengl`

Download this repository as a `.zip` and extract it, or clone the repository to an easy to find location. 

## Usage
The python executable must be in the PATH environment variable.

qtplot can be run from the commandline with `qtplot.py <filename>` or on Windows using the `qtplot.bat` file.

It is however more useful to associate `.dat` files with qtplot by selecting `qtplot.bat` in the "Open with:" menu of a `.dat` file.

`.dat` files can also be opened by dragging and dropping them on the main qtplot window.

### Main window
In the main window, the `View` tab contains a fast renderer that uses OpenGL plots the data for real-time viewing and processing. The columns that are used for plotting can be chosen with the names set in your measurement script. By clicking in the plot with the left or right mousebutton, a linecut is made and shown in the linecut window.

The `Export` tab is used to export colorplots and set the various properties that are needed. The plot can be exported in various file formats as well as to the clipboard for ease of usage.

### Linecut window
Linecuts made in the main window are plotted here. There are several controls available to make analysis easier. The updating of the ranges of the axes can be toggled with `Reset on plot`. Data can be saved to a file or the clipboard with `Data to clipboard` and `Save data...`. The figure can be copied to the clipboard with `Figure to clipboard`. The coordinate of the linecut can be included in the title of the graph with `Include Z`. 

Multiple linecuts can be added by toggling `Incremental` and setting their vertical offset. 

Markers that show the coordinates at a certain point can be set using the middle mouse button. Which coordinates to include and their significance can be set.

### Operations window
Various operations to process data are available.

### Settings window
The `.set` file that contains information about the instruments that QTLab saves next to the measurement file can be viewed using the `Settings` button in the main window. Properties of interest can be selected and then copied to the clipboard to paste in PowerPoint for example.

### Automation using the `data.py` file

By including the `data.py` file next to a personal Python script, all the core data operations and some plotting capabilities can be used. This can be useful for automating certain calculations on multiple datasets.

```
import matplotlib.pyplot as plt
from data import DatFile

df = DatFile('some_qtlab_data.dat')
# In case of a four-point measurement:
data = df.get_data('X measure', 'Y measure', 'Z measure', 'X bias', 'Y bias')
# Or a two-point measurement:
data = df.get_data('X bias', 'Y bias', 'Z measure', 'X bias', 'Y bias')

data.crop(right=-10)
data.lowpass(x_width=3, y_height=0, method='gaussian')
data.xderiv()

plt.pcolormesh(*data.get_pcolor(), cmap='seismic')
```
