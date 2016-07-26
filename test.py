import sys

from PyQt4 import QtGui, QtCore
from qtplot.qtplot import QTPlot

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        main = QTPlot(filename=sys.argv[1])
    else:
        main = QTPlot()

    sys.exit(app.exec_())
