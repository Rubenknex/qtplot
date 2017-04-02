import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT
import numpy as np
import pandas as pd
import textwrap
from PyQt4 import QtCore, QtGui, uic

from .util import eng_format, FixedOrderFormatter


class Linetrace(QtGui.QDialog):
    def __init__(self, parent, model):
        super(Linetrace, self).__init__(parent)

        self.model = model

        path = os.path.join(self.model.dir, 'ui/linetrace.ui')
        uic.loadUi(path, self)

        self.fig, self.ax = plt.subplots()

        self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter())

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('button_press_event', self.on_press)

        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)

        self.marker = None

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
        self.model.profile_changed.connect(self.set_profile)

    def get_state(self):
        return {
            'linestyle': str(self.cb_linestyle.currentText()),
            'linewidth': float(self.le_linewidth.text()),
            'marker': str(self.cb_markerstyle.currentText()),
            'markersize': float(self.le_markersize.text()),
        }

    def set_profile(self, profile):
        idx = self.cb_linestyle.findText(profile['linestyle'])
        self.cb_linestyle.setCurrentIndex(idx)
        self.le_linewidth.setText(str(profile['linewidth']))

        idx = self.cb_markerstyle.findText(profile['markerstyle'])
        self.cb_markerstyle.setCurrentIndex(idx)
        self.le_markersize.setText(str(profile['markersize']))

    def get_incremental(self):
        return self.cb_incremental.checkState() == QtCore.Qt.Checked

    def get_plot_limits(self):
        if len(self.model.linetraces) > 0:
            x, y = self.model.linetraces[-1].get_data()

            return np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)
        else:
            return 0, 1, 0, 1

    def on_pick(self, event):
        """ When a datapoint is selected, display corresponding info """
        if event.mouseevent.button == 1:
            # If more than one datapoint was found, use the middle one
            ind = event.ind[int(len(event.ind) / 2)]

            # This logic might be better placed inside the Model
            line = self.model.linetraces[-1]
            x, y = line.get_data()
            x = x[ind]
            y = y[ind]

            row = int(line.row_numbers[ind])
            data = self.model.data_file.get_row_info(row)

            # Also show the datapoint index
            data['N'] = ind

            # Fill the treeview with data
            self.tw_data.clear()
            widgets = []
            for name, value in data.items():
                if name == 'N':
                    val = str(value)
                else:
                    val = eng_format(value, 1)

                widgets.append(QtGui.QTreeWidgetItem(None, [name, val]))

            self.tw_data.insertTopLevelItems(0, widgets)

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
        # If the right mouse button is pressed in the plot, delete the
        # datapoint selection marker and clear the treewidget
        if event.button == 3:
            self.tw_data.clear()

            if self.marker is not None:
                self.marker.remove()
                self.marker = None

            self.fig.canvas.draw()

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
        self.model.clear_linetraces(redraw=True)

    def on_linetrace_changed(self, event, redraw=False, line=None):
        if event == 'add':
            x_label, y_label, z_label = line.get_labels()
            self.ax.set_xlabel(x_label)
            self.ax.set_ylabel(y_label)

            if self.cb_include_z.checkState() == QtCore.Qt.Checked:
                z = line.get_other_coord()
                title = '{0}\n{1} = {2}'.format(self.model.data_file.name,
                                                z_label, eng_format(z, 1))

            title = '\n'.join(textwrap.wrap(title, 40,
                                            replace_whitespace=False))
            self.ax.set_title(title)

            offset = float(self.le_offset.text()) * len(self.ax.lines)

            x, y = line.get_data()
            line2d = plt.Line2D(x, y + offset, color='red', picker=5,
                                **self.get_state())

            self.ax.add_line(line2d)
        elif event == 'update':
            self.ax.lines[0].set_data(*line.get_data())
        elif event == 'clear':
            while len(self.ax.lines) > 0:
                self.ax.lines[-1].remove()

        if redraw:
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

            # This results in speed up but also event handling errors
            #self.fig.canvas.flush_events()

    def show_window(self):
        self.show()
        self.raise_()

    def closeEvent(self, event):
        self.hide()
        event.ignore()
