import sys

from PyQt4 import QtGui, QtCore
from qtplot.qtplot import QTPlot

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        main = QTPlot(filename='C:\\Users\\LocalAdmin\\Dropbox\\QuTech\\data\\dev8_930.dat')
    else:
        main = QTPlot()

    #main = QTPlot(filename='C:\\Users\\LocalAdmin\\Dropbox\\QuTech\\data\\dev8_930.dat')

    sys.exit(app.exec_())
