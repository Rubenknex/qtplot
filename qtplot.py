import sys

from PyQt4 import QtGui

from qtplot.main_window import MainWindow

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)

    if len(sys.argv) > 1:
        main = MainWindow(filename=sys.argv[1])
    else:
        main = MainWindow()

    sys.exit(app.exec_())
