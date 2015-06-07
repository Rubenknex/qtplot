import matplotlib.pyplot as plt
import numpy as np

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.ticker import ScalarFormatter
from scipy.spatial import qhull, delaunay_plot_2d
from PyQt4 import QtGui, QtCore

class FixedOrderFormatter(ScalarFormatter):
    """Format numbers using engineering notation."""
    def __init__(self, significance=0):
        ScalarFormatter.__init__(self, useOffset=None, useMathText=None)
        self.format = '%.' + str(significance) + 'f'

    def __call__(self, x, pos=None):
        if x == 0:
            return '0'

        exp = self.orderOfMagnitude

        return self.format % (x / (10 ** exp))

    def _set_orderOfMagnitude(self, range):
        exp = np.floor(np.log10(range))
        self.orderOfMagnitude = exp - (exp % 3)

class ExportWidget(QtGui.QWidget):
	def __init__(self, main):
		QtGui.QWidget.__init__(self)

		self.main = main

		self.fig, self.ax = plt.subplots()
		self.cb = None

		self.canvas = FigureCanvasQTAgg(self.fig)
		self.toolbar = NavigationToolbar2QT(self.canvas, self)

		self.b_update = QtGui.QPushButton('Update', self)
		self.b_update.clicked.connect(self.on_update)

		grid = QtGui.QGridLayout()

		grid.addWidget(QtGui.QLabel('Title'), 1, 1)
		self.le_title = QtGui.QLineEdit('test')
		grid.addWidget(self.le_title, 1, 2)



		
		grid.addWidget(QtGui.QLabel('X Label'), 2, 1)
		self.le_x_label = QtGui.QLineEdit('test')
		grid.addWidget(self.le_x_label, 2, 2)

		grid.addWidget(QtGui.QLabel('X Format'), 2, 3)
		self.le_x_format = QtGui.QLineEdit('test')
		grid.addWidget(self.le_x_format, 2, 4)

		grid.addWidget(QtGui.QLabel('Y Label'), 3, 1)
		self.le_y_label = QtGui.QLineEdit('test')
		grid.addWidget(self.le_y_label, 3, 2)

		grid.addWidget(QtGui.QLabel('Y Format'), 3, 3)
		self.le_y_format = QtGui.QLineEdit('test')
		grid.addWidget(self.le_y_format, 3, 4)

		grid.addWidget(QtGui.QLabel('Z Label'), 4, 1)
		self.le_z_label = QtGui.QLineEdit('test')
		grid.addWidget(self.le_z_label, 4, 2)

		grid.addWidget(QtGui.QLabel('Z Format'), 4, 3)
		self.le_z_format = QtGui.QLineEdit('test')
		grid.addWidget(self.le_z_format, 4, 4)

		vbox = QtGui.QVBoxLayout(self)
		vbox.addWidget(self.toolbar)
		vbox.addWidget(self.canvas)
		vbox.addWidget(self.b_update)
		vbox.addLayout(grid)

	def set_info(self, title, x, y, z):
		self.le_title.setText(title)
		self.le_x_label.setText(x)
		self.le_y_label.setText(y)
		self.le_z_label.setText(z)

	def on_update(self):
		if self.main.data != None:
			self.ax.clear()

			x, y, z = self.main.data.get_pcolor()
			quadmesh = self.ax.pcolormesh(x, y, z, cmap='seismic')
			quadmesh.get_cmap().set_gamma(self.main.canvas.colormap.gamma)
			quadmesh.set_clim(self.main.canvas.colormap.get_limits())
			self.ax.axis('tight')

			self.ax.set_title(self.le_title.text())
			self.ax.set_xlabel(self.le_x_label.text())
			self.ax.set_ylabel(self.le_y_label.text())
			
			self.ax.xaxis.set_major_formatter(FixedOrderFormatter())
			self.ax.yaxis.set_major_formatter(FixedOrderFormatter())

			if self.cb == None:
				self.cb = self.fig.colorbar(quadmesh)
			else:
				self.cb.set_clim(quadmesh.get_clim())
			
			self.main.data.gen_delaunay()
			if self.main.data.tri != None:
				print 'plotting triang'
				delaunay_plot_2d(self.main.data.tri, self.ax)

			self.cb.set_label(self.le_z_label.text())
			self.cb.draw_all()

			self.canvas.draw()