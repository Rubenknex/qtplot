import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT
import numpy as np
import pandas as pd
from PyQt4 import QtCore, QtGui, uic


class Linetrace(QtGui.QDialog):
    def __init__(self, parent, model):
        super(Linetrace, self).__init__(parent)

        self.model = model

        path = os.path.join(self.model.dir, 'ui/linetrace.ui')
        uic.loadUi(path, self)

        self.fig, self.ax = plt.subplots()

        self.canvas = FigureCanvasQTAgg(self.fig)
        #self.canvas.mpl_connect('pick_event', self.on_pick)
        #self.canvas.mpl_connect('button_press_event', self.on_press)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)

        self.tw_data.setHidden(True)

        self.bind()

        pos = parent.mapToGlobal(parent.rect().topRight())
        self.move(pos.x() + 3, pos.y() - 25)

        self.show()

    def bind(self):
        self.b_copy_data.clicked.connect(self.on_data_to_clipboard)
        self.b_copy_figure.clicked.connect(self.on_figure_to_clipboard)
        self.b_to_ppt.clicked.connect(self.on_to_ppt)
        self.b_save_data.clicked.connect(self.on_save)
        self.b_toggle_info.clicked.connect(self.on_toggle_datapoint_info)
        self.b_clear.clicked.connect(self.on_clear)

        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"),
                        self, self.on_figure_to_clipboard)

        self.model.linetrace_changed.connect(self.on_linetrace_changed)

    def get_state(self):
        return {
            'linestyle': str(self.cb_linestyle.currentText()),
            'linewidth': float(self.le_linewidth.text()),
            'marker': str(self.cb_markerstyle.currentText()),
            'markersize': float(self.le_markersize.text()),
        }

    def set_state(self, profile):
        idx = self.cb_linestyle.findText(profile['linestyle'])
        self.cb_linestyle.setCurrentIndex(idx)
        self.le_linewidth.setText(profile['linewidth'])

        idx = self.cb_markerstyle.findText(profile['markerstyle'])
        self.cb_markerstyle.setCurrentIndex(idx)
        self.le_markersize.setText(profile['markersize'])

    def get_plot_limits(self):
        if self.model.linetraces:
            # use matplotlib data instead of model
            x = np.concatenate(tuple(line.x for line in self.model.linetraces))
            y = np.concatenate(tuple(line.y for line in self.model.linetraces))
        else:
            x = [0, 1]
            y = [0, 1]

        return np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)

    def on_data_to_clipboard(self):
        if self.x is None or self.y is None:
            return

        data = pd.DataFrame(np.column_stack((self.x, self.y)),
                            columns=[self.xlabel, self.ylabel])

        data.to_clipboard(index=False)

    def on_figure_to_clipboard(self):
        path = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(path, 'test.png')
        self.fig.savefig(path, bbox_inches='tight')

        img = QtGui.QImage(path)
        QtGui.QApplication.clipboard().setImage(img)

    def on_to_ppt(self):
        """ Some win32 COM magic to interact with powerpoint """
        try:
            import win32com.client
        except ImportError:
            print('ERROR: The win32com library needs to be installed')
            return

        # First, copy to the clipboard
        self.on_figure_to_clipboard()

        # Connect to an open PowerPoint application
        app = win32com.client.Dispatch('PowerPoint.Application')

        # Get the current slide and paste the plot
        slide = app.ActiveWindow.View.Slide
        shape = slide.Shapes.Paste()

        # Add a hyperlink to the data location to easily open the data again
        shape.ActionSettings[0].Hyperlink.Address = self.main.abs_filename

    def on_save(self):
        if self.x is None or self.y is None:
            return

        path = os.path.dirname(os.path.realpath(__file__))
        filename = QtGui.QFileDialog.getSaveFileName(self,
                                                     'Save file',
                                                     path,
                                                     '.dat')

        if filename != '':
            data = pd.DataFrame(np.column_stack((self.x, self.y)),
                                columns=[self.xlabel, self.ylabel])

            data.to_csv(filename, sep='\t', index=False)

    def on_toggle_datapoint_info(self):
        self.tw_data.setHidden(not self.tw_data.isHidden())

    def on_clear(self):
        while len(self.ax.lines) > 0:
            self.ax.lines.pop(0)

        self.model.linetraces = []

        self.fig.canvas.draw()

    def on_linetrace_changed(self, event, linetrace=None):
        if event == 'add':
            if self.cb_incremental.checkState() == QtCore.Qt.Unchecked:
                # Delete all existing lines
                while len(self.ax.lines) > 0:
                    self.ax.lines.pop(0)

                del self.model.linetraces[:-1]

            offset = float(self.le_offset.text()) * len(self.ax.lines)

            line = plt.Line2D(linetrace.x, linetrace.y + offset,
                              color='red', **self.get_state())

            self.ax.add_line(line)

        # Add some extra space to the plot limits
        if self.cb_reset_on_plot.checkState() == QtCore.Qt.Checked:
            minx, maxx, miny, maxy = self.get_plot_limits()

            xdiff = (maxx - minx) * .05
            ydiff = (maxy - miny) * .05

            self.ax.axis([minx - xdiff, maxx + xdiff,
                          miny - ydiff, maxy + ydiff])

        self.ax.set_aspect('auto')
        self.fig.tight_layout()

        self.fig.canvas.draw()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
