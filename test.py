import sys
import os

#from PyQt4 import QtGui, QtCore
#from qtpy imbport QtWidgets, QtCore
#from qtplot.qtplot import QTPlot
from qtplot import qtplot
from qtplot import controller

if __name__ == '__main__':
    #qtplot.main()
    controller.main()
    """
    app = QtGui.QApplication(sys.argv)

    print(os.path.dirname(os.path.realpath(__file__)))

    #if len(sys.argv) > 1:
    #    main = QTPlot(filename='C:\\Users\\LocalAdmin\\Dropbox\\QuTech\\data\\dev8_930.dat')
    #else:
    #    main = QTPlot()

    #main = QTPlot(filename='C:\\Users\\LocalAdmin\\Dropbox\\QuTech\\data\\dev8_930.dat')
    main = QTPlot()

    sys.exit(app.exec_())
    """
