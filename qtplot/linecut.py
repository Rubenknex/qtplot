import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import textwrap
from itertools import cycle

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT

from PyQt4 import QtGui, QtCore

from .util import FixedOrderFormatter, eng_format


class Linetrace(plt.Line2D):
    """
    Represents a linetrace from the data. The purpose of this class is
    to be able to store incremental linetraces in an array.

    x/y:        Arrays containing x and y data
    type:       Type of linetrace, 'horizontal' or 'vertical'
    position:   The x/y coordinate at which the linetrace was taken
    """

    def __init__(self, x, y, row_numbers, type, position, **kwargs):
        plt.Line2D.__init__(self, x, y, **kwargs)

        self.row_numbers = row_numbers
        self.type = type
        self.position = position


class Linecut(QtGui.QDialog):
    def __init__(self, main=None):
        super(Linecut, self).__init__(None)

        self.main = main

        self.fig, self.ax = plt.subplots()
        self.x, self.y = None, None
        self.linetraces = []
        self.marker = None
        self.colors = cycle('bgrcmykw')

        self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter())

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Linecut")
        # Don't show this window in the taskbar
        self.setWindowFlags(QtCore.Qt.Tool)

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        hbox_export = QtGui.QHBoxLayout()

        self.cb_reset_cmap = QtGui.QCheckBox('Reset on plot')
        self.cb_reset_cmap.setCheckState(QtCore.Qt.Checked)
        hbox_export.addWidget(self.cb_reset_cmap)

        self.b_save = QtGui.QPushButton('Copy data', self)
        self.b_save.clicked.connect(self.on_data_to_clipboard)
        hbox_export.addWidget(self.b_save)

        self.b_copy = QtGui.QPushButton('Copy figure', self)
        self.b_copy.clicked.connect(self.on_figure_to_clipboard)
        QtGui.QShortcut(QtGui.QKeySequence("Ctrl+C"),
                        self, self.on_figure_to_clipboard)
        hbox_export.addWidget(self.b_copy)

        self.b_to_ppt = QtGui.QPushButton('To PPT (Win)', self)
        self.b_to_ppt.clicked.connect(self.on_to_ppt)
        hbox_export.addWidget(self.b_to_ppt)

        self.b_save_dat = QtGui.QPushButton('Save data...', self)
        self.b_save_dat.clicked.connect(self.on_save)
        hbox_export.addWidget(self.b_save_dat)

        self.b_toggle_info = QtGui.QPushButton('Toggle info')
        self.b_toggle_info.clicked.connect(self.on_toggle_datapoint_info)
        hbox_export.addWidget(self.b_toggle_info)

        # Linecuts
        hbox_linecuts = QtGui.QHBoxLayout()

        hbox_linecuts.addWidget(QtGui.QLabel('Linecuts'))

        self.cb_incremental = QtGui.QCheckBox('Incremental')
        self.cb_incremental.setCheckState(QtCore.Qt.Unchecked)
        hbox_linecuts.addWidget(self.cb_incremental)

        hbox_linecuts.addWidget(QtGui.QLabel('Offset:'))

        self.le_offset = QtGui.QLineEdit('0', self)
        hbox_linecuts.addWidget(self.le_offset)

        self.b_clear_lines = QtGui.QPushButton('Clear', self)
        self.b_clear_lines.clicked.connect(self.on_clear_lines)
        hbox_linecuts.addWidget(self.b_clear_lines)

        # Lines
        hbox_style = QtGui.QHBoxLayout()

        hbox_style.addWidget(QtGui.QLabel('Line style'))
        self.cb_linestyle = QtGui.QComboBox(self)
        self.cb_linestyle.addItems(['None', 'solid', 'dashed', 'dotted'])
        hbox_style.addWidget(self.cb_linestyle)

        hbox_style.addWidget(QtGui.QLabel('Linewidth'))
        self.le_linewidth = QtGui.QLineEdit('0.5', self)
        hbox_style.addWidget(self.le_linewidth)

        # Markers
        hbox_style.addWidget(QtGui.QLabel('Marker style'))
        self.cb_markerstyle = QtGui.QComboBox(self)
        self.cb_markerstyle.addItems(['None', '.', 'o', 'x'])
        hbox_style.addWidget(self.cb_markerstyle)

        hbox_style.addWidget(QtGui.QLabel('Size'))
        self.le_markersize = QtGui.QLineEdit('0.5', self)
        hbox_style.addWidget(self.le_markersize)

        self.cb_include_z = QtGui.QCheckBox('Include Z')
        self.cb_include_z.setCheckState(QtCore.Qt.Checked)
        hbox_style.addWidget(self.cb_include_z)

        self.row_tree = QtGui.QTreeWidget(self)
        self.row_tree.setHeaderLabels(['Parameter', 'Value'])
        self.row_tree.setColumnWidth(0, 100)
        self.row_tree.setHidden(True)

        hbox_plot = QtGui.QHBoxLayout()
        hbox_plot.addWidget(self.canvas)
        hbox_plot.addWidget(self.row_tree)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addLayout(hbox_plot)
        layout.addLayout(hbox_export)
        layout.addLayout(hbox_linecuts)
        layout.addLayout(hbox_style)
        self.setLayout(layout)

        self.resize(700, 500)
        self.move(630, 100)

    def populate_ui(self):
        profile = self.main.profile_settings

        idx = self.cb_linestyle.findText(profile['line_style'])
        self.cb_linestyle.setCurrentIndex(idx)
        self.le_linewidth.setText(profile['line_width'])

        idx = self.cb_markerstyle.findText(profile['marker_style'])
        self.cb_markerstyle.setCurrentIndex(idx)
        self.le_markersize.setText(profile['marker_size'])

    def get_line_kwargs(self):
        return {
            'linestyle': str(self.cb_linestyle.currentText()),
            'linewidth': float(self.le_linewidth.text()),
            'marker': str(self.cb_markerstyle.currentText()),
            'markersize': float(self.le_markersize.text()),
        }

    def on_reset(self):
        if self.x is not None and self.y is not None:
            minx, maxx = np.min(self.x), np.max(self.x)
            miny, maxy = np.min(self.y), np.max(self.y)

            xdiff = (maxx - minx) * .1
            ydiff = (maxy - miny) * .1

            self.ax.axis([minx - xdiff, maxx + xdiff,
                          miny - ydiff, maxy + ydiff])

            self.canvas.draw()

    def on_pick(self, event):
        if event.mouseevent.button == 1:
            line = self.linetraces[0]

            ind = event.ind[int(len(event.ind) / 2)]
            x = line.get_xdata()[ind]
            y = line.get_ydata()[ind]

            row = int(line.row_numbers[ind])
            data = self.main.dat_file.get_row_info(row)

            # Also show the datapoint index
            data['N'] = ind

            # Fill the treeview with data
            self.row_tree.clear()
            widgets = []
            for name, value in data.items():
                if name == 'N':
                    val = str(value)
                else:
                    val = eng_format(value, 1)

                widgets.append(QtGui.QTreeWidgetItem(None, [name, val]))

            self.row_tree.insertTopLevelItems(0, widgets)

            # Remove the previous datapoint marker
            if self.marker is not None:
                self.marker.remove()
                self.marker = None

            # Plot a new datapoint marker
            self.marker = self.ax.plot(x, y, '.',
                                       markersize=15,
                                       color='black')[0]

        self.fig.canvas.draw()

    def on_press(self, event):
        if event.button == 3:
            self.row_tree.clear()

            if self.marker is not None:
                self.marker.remove()
                self.marker = None

            self.fig.canvas.draw()

    def on_toggle_datapoint_info(self):
        self.row_tree.setHidden(not self.row_tree.isHidden())

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

    def on_clear_lines(self):
        for line in self.linetraces:
            line.remove()

        self.linetraces = []

        self.fig.canvas.draw()

    def plot_linetrace(self, x, y, z, row_numbers, type, position, title,
                       xlabel, ylabel, otherlabel):
        # Don't draw lines consisting of one point
        if np.count_nonzero(~np.isnan(y)) < 2:
            return

        self.xlabel, self.ylabel, self.otherlabel = xlabel, ylabel, otherlabel
        self.title = title
        self.x, self.y, self.z = x, y, z

        if self.cb_include_z.checkState() == QtCore.Qt.Checked:
            title = '{0}\n{1} = {2}'.format(title, otherlabel, eng_format(z, 1))

        title = '\n'.join(textwrap.wrap(title, 40, replace_whitespace=False))
        self.ax.set_title(title)

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # Remove all the existing lines and only plot one if we uncheck
        # the incremental box. Else, add a new line to the collection
        if self.cb_incremental.checkState() == QtCore.Qt.Unchecked:
            for line in self.linetraces:
                line.remove()

            self.linetraces = []

            line = Linetrace(x, y, row_numbers, type, position,
                             color='red',
                             picker=5,
                             **self.get_line_kwargs())

            self.linetraces.append(line)
            self.ax.add_line(line)

            self.total_offset = 0
        else:
            if len(self.ax.lines) > 0:
                if self.linetraces[-1].position == position:
                    return

            index = len(self.linetraces) - 1

            offset = float(self.le_offset.text())
            line = Linetrace(x, y + index * offset, row_numbers, type, position)
            line.set_color(next(self.colors))

            self.linetraces.append(line)
            self.ax.add_line(line)

        if self.cb_reset_cmap.checkState() == QtCore.Qt.Checked:
            x, y = np.ma.masked_invalid(x), np.ma.masked_invalid(y)
            minx, maxx = np.min(x), np.max(x)
            miny, maxy = np.min(y), np.max(y)

            xdiff = (maxx - minx) * .05
            ydiff = (maxy - miny) * .05

            self.ax.axis([minx - xdiff, maxx + xdiff,
                          miny - ydiff, maxy + ydiff])

        self.ax.set_aspect('auto')
        self.fig.tight_layout()

        self.fig.canvas.draw()

    def resizeEvent(self, event):
        self.fig.tight_layout()
        self.canvas.draw()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
