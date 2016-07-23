import matplotlib as mpl
import sys

from PyQt4 import QtGui

from qtplot.main_window import MainWindow
from qtplot.linecut import Linecut
from qtplot.operations import Operations

if __name__ == '__main__':
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        main = MainWindow(filename=sys.argv[1])
    else:
        main = MainWindow()

    sys.exit(app.exec_())
