import matplotlib as mpl
import matplotlib.pyplot as plt
import textwrap

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg, \
    NavigationToolbar2QT
from PyQt4 import QtGui, QtCore, uic

from .util import FixedOrderFormatter
import os


class Export(QtGui.QWidget):
    def __init__(self, parent, model):
        super(Export, self).__init__(parent)

        self.model = model

        path = os.path.join(self.model.dir, 'ui/export.ui')
        uic.loadUi(path, self)

        self.fig, self.ax = plt.subplots()
        self.cb = None

        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.plot_layout.addWidget(self.toolbar)
        self.plot_layout.addWidget(self.canvas)

        self.bind()

        """
        # Set some matplotlib font settings
        mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
        mpl.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
        mpl.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
        """

    def bind(self):
        """ Set up connections for GUI and model events """
        self.b_update.clicked.connect(self.on_update)
        self.b_copy.clicked.connect(self.on_copy)
        self.b_to_ppt.clicked.connect(self.on_to_ppt)
        self.b_export.clicked.connect(self.on_export)

        self.model.profile_changed.connect(self.set_profile)

    def set_profile(self, profile):
        """ Update the interface with the profile settings """
        self.le_title.setText(profile['title'])
        #self.le_dpi.setText(str(profile['dpi']))
        self.cb_rasterize.setChecked(bool(profile['rasterize']))

        self.le_x_label.setText(profile['x_label'])
        self.le_y_label.setText(profile['y_label'])
        self.le_z_label.setText(profile['z_label'])

        self.le_x_format.setText(profile['x_format'])
        self.le_y_format.setText(profile['y_format'])
        self.le_z_format.setText(profile['z_format'])

        self.le_x_div.setText(profile['x_div'])
        self.le_y_div.setText(profile['y_div'])
        self.le_z_div.setText(profile['z_div'])

        self.le_font.setText(profile['font'])
        self.le_width.setText(str(profile['width']))
        # cb orient

        self.le_font_size.setText(str(profile['font_size']))
        self.le_height.setText(str(profile['height']))
        # cb pos

        self.cb_triangulation.setChecked(bool(profile['triangulation']))
        self.cb_tripcolor.setChecked(bool(profile['tripcolor']))
        self.cb_linetrace.setChecked(bool(profile['linetrace']))

    def get_state(self):
        """ Return all properties that are to be saved in a profile """
        state = {
            'title': str(self.le_title.text()),
            'rasterize': bool(self.cb_rasterize.isChecked()),
            'x_label': str(self.le_x_label.text()),
            'y_label': str(self.le_y_label.text()),
            'z_label': str(self.le_z_label.text()),
            'x_format': str(self.le_x_format.text()),
            'y_format': str(self.le_y_format.text()),
            'z_format': str(self.le_z_format.text()),
            'x_div': str(self.le_x_div.text()),
            'y_div': str(self.le_y_div.text()),
            'z_div': str(self.le_z_div.text()),
            'font': str(self.le_font.text()),
            'font_size': int(self.le_font_size.text()),
            'width': float(self.le_width.text()),
            'height': float(self.le_height.text()),
            'triangulation': bool(self.cb_triangulation.isChecked()),
            'tripcolor': bool(self.cb_tripcolor.isChecked()),
            'linetrace': bool(self.cb_linetrace.isChecked()),
        }

        return state

    def keyPressEvent(self, e):
        """ When return is pressed inside the window, update the plot """
        if e.key() == QtCore.Qt.Key_Return:
            self.on_update()

    def format_label(self, s):
        """ Some preset macros are replaced by properties of the data """
        conversions = {
            '<filename>': self.model.data2d.filename,
            '<x>': self.model.data2d.x_name,
            '<y>': self.model.data2d.y_name,
            '<z>': self.model.data2d.z_name
        }

        for old, new in conversions.items():
            s = s.replace(old, new)

        return s

    def on_update(self):
        """ Draw the entire plot """
        if self.model.data2d is None:
            return

        data = self.model.data2d

        font = {
            'family': str(self.le_font.text()),
            'size': int(str(self.le_font_size.text()))
        }

        mpl.rc('font', **font)

        # Clear the plot
        self.ax.clear()

        # Get the data and colormap
        x, y, z = data.get_pcolor()
        cmap = self.model.colormap.get_mpl_colormap()

        tri_checkboxes = [self.cb_tripcolor.checkState(),
                          self.cb_triangulation.checkState()]

        # If we are going to need to plot triangulation data, prepare
        # the data so it can be plotted
        if QtCore.Qt.Checked in tri_checkboxes:
            if data.tri is None:
                data.generate_triangulation()

            xc, yc = data.get_triangulation_coordinates()

            tri = mpl.tri.Triangulation(xc, yc, data.tri.simplices)

        # Plot the data using either pcolormesh or tripcolor
        if self.cb_tripcolor.checkState() != QtCore.Qt.Checked:
            quadmesh = self.ax.pcolormesh(x, y, z,
                                          cmap=cmap,
                                          rasterized=True)

            quadmesh.set_clim(self.model.colormap.get_limits())
        else:
            quadmesh = self.ax.tripcolor(tri, data.z.ravel(),
                                         cmap=cmap, rasterized=True)

            quadmesh.set_clim(self.model.colormap.get_limits())

        # Plot the triangulation
        if self.cb_triangulation.checkState() == QtCore.Qt.Checked:
            self.ax.triplot(tri, 'o-', color='black',
                            linewidth=0.5, markersize=3)

        self.ax.axis('tight')

        title = self.format_label(str(self.le_title.text()))
        title = '\n'.join(textwrap.wrap(title, 40,
                                        replace_whitespace=False))

        # Set all the plot labels
        self.ax.set_title(title)
        self.ax.set_xlabel(self.format_label(self.le_x_label.text()))
        self.ax.set_ylabel(self.format_label(self.le_y_label.text()))

        # Set the axis tick formatters
        self.ax.xaxis.set_major_formatter(FixedOrderFormatter(
            str(self.le_x_format.text()), float(self.le_x_div.text())))
        self.ax.yaxis.set_major_formatter(FixedOrderFormatter(
            str(self.le_y_format.text()), float(self.le_y_div.text())))

        if self.cb is not None:
            self.cb.remove()

        # Colorbar layout
        orientation = str(self.cb_cb_orient.currentText())
        self.cb = self.fig.colorbar(quadmesh, orientation=orientation)

        self.cb.formatter = FixedOrderFormatter(
            str(self.le_z_format.text()), float(self.le_z_div.text()))

        self.cb.update_ticks()

        self.cb.set_label(self.format_label(self.le_z_label.text()))
        self.cb.draw_all()

        # Plot the current linecut if neccesary
        if self.cb_linetrace.checkState() == QtCore.Qt.Checked:
            for line in self.model.linetraces:
                pass
                """
                if line.type == 'horizontal':
                    plt.axhline(line.position, color='red')
                elif line.type == 'vertical':
                    plt.axvline(linetrace.position, color='red')
                """

        self.fig.tight_layout()

        self.canvas.draw()

    def on_copy(self):
        """ Copy the current plot to the clipboard """
        path = os.path.join(self.model.dir, 'test.png')
        self.fig.savefig(path)

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
        self.on_copy()

        # Connect to an open PowerPoint application
        app = win32com.client.Dispatch('PowerPoint.Application')

        # Get the current slide and paste the plot
        slide = app.ActiveWindow.View.Slide
        shape = slide.Shapes.Paste()

        # Add a hyperlink to the data location to easily open the data again
        shape.ActionSettings[0].Hyperlink.Address = self.main.abs_filename

    def on_export(self):
        """ Export the current plot to a file """
        filters = ('Portable Network Graphics (*.png);;'
                   'Portable Document Format (*.pdf);;'
                   'Postscript (*.ps);;'
                   'Encapsulated Postscript (*.eps);;'
                   'Scalable Vector Graphics (*.svg)')

        filename = QtGui.QFileDialog.getSaveFileName(self,
                                                     caption='Export figure',
                                                     directory=self.model.dir,
                                                     filter=filters)
        filename = str(filename)

        if filename != '':
            previous_size = self.fig.get_size_inches()
            self.fig.set_size_inches(float(self.le_width.text()),
                                     float(self.le_height.text()))

            #dpi = int(self.le_dpi.text())

            self.fig.savefig(filename, bbox_inches='tight')
            self.fig.set_size_inches(previous_size)

            self.canvas.draw()
