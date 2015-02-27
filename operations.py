import numpy as np
import pandas as pd
from PyQt4 import QtGui, QtCore
from scipy import ndimage

def op_abs(data, op, **kwargs):
    return data.applymap(np.absolute)

def op_autoflip(data, op, **kwargs):
    if data.index[0] > data.index[-1]:
        data = data.reindex(index=data.index[::-1])
    if data.columns[0] > data.columns[-1]:
        data = data.reindex(columns=data.columns[::-1])
    
    return data

def op_crop(data, op, **kwargs):
    x1, x2 = op.get_property('Left', int), op.get_property('Right', int)
    y1, y2 = op.get_property('Bottom', int), op.get_property('Top', int)

    return data.iloc[x1:x2, y1:y2]

def op_dderiv(data, op, **kwargs):
    xcomp = op_xderiv(data, op)
    ycomp = op_yderiv(data, op)

    xvalues = np.delete(xcomp.values, -1, axis=0)
    yvalues = np.delete(ycomp.values, -1, axis=1)

    theta = np.radians(op.get_property('Theta', float))
    xdir, ydir = np.cos(theta), np.sin(theta)

    return pd.DataFrame(xvalues * xdir + yvalues * ydir, ycomp.index, xcomp.columns)

def op_equalize(data, op, **kwargs):
    return data

def op_even_odd(data, op, **kwargs):
    return data

def op_flip(data, op, **kwargs):
    if op.get_property('X Axis'):
        data = data.reindex(columns=data.columns[::-1])

    if op.get_property('Y Axis'):
        data = data.reindex(index=data.index[::-1])

    return data

def op_gradmag(data, op, **kwargs):
    xcomp = op_xderiv(data, op)
    ycomp = op_yderiv(data, op)

    xvalues = np.delete(xcomp.values, -1, axis=0)
    yvalues = np.delete(ycomp.values, -1, axis=1)

    return pd.DataFrame(np.sqrt(xvalues**2 + yvalues**2), ycomp.index, xcomp.columns)

def op_highpass(data, op, **kwargs):
    sx, sy = op.get_property('X Width', float), op.get_property('Y Height', float)
    values = ndimage.filters.gaussian_filter(data.values, [sy, sx])

    return pd.DataFrame(data.values - values, data.index, data.columns)

def op_hist2d(data, op, **kwargs):
    return data

def op_log(data, op, **kwargs):
    return pd.DataFrame(np.log10(data.values), data.index, data.columns)

def op_lowpass(data, op, **kwargs):
    # X and Y sigma order?
    sx, sy = op.get_property('X Width', float), op.get_property('Y Height', float)
    values = ndimage.filters.gaussian_filter(data.values, [sy, sx])

    return pd.DataFrame(values, data.index, data.columns)

def op_neg(data, op, **kwargs):
    return data.applymap(np.negative)

def op_normalize(data, op, **kwargs):
    return data

def op_offset(data, op, **kwargs):
    offset = op.get_property('Offset', float)

    return pd.DataFrame(data.values + offset, data.index, data.columns)

def op_offset_axes(data, op, **kwargs):
    x_off, y_off = op.get_property('X Offset', float), op.get_property('Y Offset', float)

    return pd.DataFrame(data.values, data.index + y_off, data.columns + x_off)

def op_power(data, op, **kwargs):
    power = op.get_property('Power', float)

    return pd.DataFrame(np.power(data.values, power), data.index, data.columns)

def op_rotate_ccw(data, op, **kwargs):
    return data

def op_rotate_cw(data, op, **kwargs):
    return pd.DataFrame(data.values, data.index, data.columns)

def op_scale_axes(data, op, **kwargs):
    x_sc, y_sc = op.get_property('X Scale', float), op.get_property('Y Scale', float)

    return pd.DataFrame(data.values, data.index * y_sc, data.columns * x_sc)

def op_scale_data(data, op, **kwargs):
    factor = op.get_property('Factor', float)

    return pd.DataFrame(data.values * factor, data.index, data.columns)

def op_sub_linecut(data, op, **kwargs):
    linecut_type = kwargs['linecut_type']
    linecut_coord = kwargs['linecut_coord']

    if linecut_type == None:
        return data

    if linecut_type == 'horizontal':
        lc_data = np.array(data.loc[linecut_coord])
    elif linecut_type == 'vertical':
        lc_data = np.array(data[linecut_coord])
        lc_data = lc_data[:,np.newaxis]

    return pd.DataFrame(data.values - lc_data, data.index, data.columns)

def op_xderiv(data, op, **kwargs):
    dx = np.diff(data.columns)
    ddata = np.diff(data.values, axis=1)

    return pd.DataFrame(ddata / dx, data.index, data.columns[:-1] + dx)

def op_yderiv(data, op, **kwargs):
    dy = np.diff(data.index)
    ddata = np.diff(data.values, axis=0)

    return pd.DataFrame(ddata / dy[np.newaxis,:].T, data.index[:-1] + dy, data.columns)

class Operation(QtGui.QWidget):
    def __init__(self, name, func, widgets=[]):
        super(Operation, self).__init__(None)

        layout = QtGui.QGridLayout(self)
        self.name = name
        self.func = func
        self.items = {}

        height = 1

        for widget in widgets:
            typ, name, data = widget

            if typ == 'checkbox':
                checkbox = QtGui.QCheckBox(name)
                checkbox.setChecked(data)
                layout.addWidget(checkbox, height, 2)

                self.items[name] = checkbox
            elif typ == 'textbox':
                lineedit = QtGui.QLineEdit(data)
                layout.addWidget(QtGui.QLabel(name), height, 1)
                layout.addWidget(lineedit, height, 2)

                self.items[name] = lineedit
            elif typ == 'combobox':
                layout.addWidget(QtGui.QLabel(name), height, 1)
                combobox = QtGui.QComboBox()
                combobox.addItems(data)
                layout.addWidget(combobox, height, 2)

                self.items[name] = combobox

            height += 1

        print self.get_formatted()

    def get_property(self, name, cast=None):
        if name in self.items:
            widget = self.items[name]

            if type(widget) is QtGui.QCheckBox:
                return widget.isChecked()
            elif type(widget) is QtGui.QLineEdit:
                return cast(str(widget.text()))
            elif type(widget) is QtGui.QComboBox:
                return str(widget.currentText())
        else:
            raise Exception('Operation doesn\'t have the property: ' + name)

    def get_formatted(self):
        params = self.name + ';'

        for name in self.items:
            widget = self.items[name]

            params += name + ':'
            if type(widget) is QtGui.QCheckBox:
                params += str(widget.isChecked())
            elif type(widget) is QtGui.QLineEdit:
                params += str(widget.text())
            elif type(widget) is QtGui.QComboBox:
                params += str(widget.currentText())
            params += ';'

        return params

class Operations(QtGui.QDialog):
    def __init__(self, parent=None):
        super(Operations, self).__init__(parent)

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Operations")

        self.items = {
            'abs':          [op_abs],
            'autoflip':     [op_autoflip],
            'crop':         [op_crop, [('textbox', 'Left', '0'), ('textbox', 'Right', '0'), ('textbox', 'Bottom', '0'), ('textbox', 'Top', '0')]],
            'dderiv':       [op_dderiv, [('textbox', 'Theta', '0')]],
            'equalize':     [op_equalize],
            'even odd':     [op_even_odd],
            'flip':         [op_flip, [('checkbox', 'X Axis', False), ('checkbox', 'Y Axis', False)]],
            'gradmag':      [op_gradmag],
            'highpass':     [op_highpass, [('textbox', 'X Width', '3'), ('textbox', 'Y Height', '3'), ('combobox', 'Type', ['Gaussian', 'Lorentzian', 'Exponential', 'Thermal'])]],
            'hist2d':       [op_hist2d],
            'log':          [op_log],
            'lowpass':      [op_lowpass, [('textbox', 'X Width', '3'), ('textbox', 'Y Height', '3'), ('combobox', 'Type', ['Gaussian', 'Lorentzian', 'Exponential', 'Thermal'])]],
            'neg':          [op_neg],
            'normalize':    [op_normalize],
            'offset':       [op_offset, [('textbox', 'Offset', '0')]],
            'offset axes':  [op_offset_axes],
            'power':        [op_power, [('textbox', 'Power', '1')]],
            'rotate ccw':   [op_rotate_ccw],
            'rotate cw':    [op_rotate_cw],
            'scale axes':   [op_scale_axes],
            'scale data':   [op_scale_data, [('textbox', 'Factor', '1')]],
            'sub linecut':  [op_sub_linecut],
            'xderiv':       [op_xderiv],
            'yderiv':       [op_yderiv],
        }

        self.options = QtGui.QListWidget(self)
        self.options.addItems(sorted(self.items.keys()))

        self.b_add = QtGui.QPushButton('Add')
        self.b_add.clicked.connect(self.on_add)

        self.b_up = QtGui.QPushButton('Up')
        self.b_up.clicked.connect(self.on_up)

        self.b_down = QtGui.QPushButton('Down')
        self.b_down.clicked.connect(self.on_down)

        self.b_remove = QtGui.QPushButton('Remove')
        self.b_remove.clicked.connect(self.on_remove)

        self.b_clear = QtGui.QPushButton('Clear')
        self.b_clear.clicked.connect(self.on_clear)

        self.b_update = QtGui.QPushButton('Update')
        self.b_update.clicked.connect(self.on_update)

        self.b_load = QtGui.QPushButton('Load...')
        self.b_load.clicked.connect(self.on_load)

        self.b_save = QtGui.QPushButton('Save...')
        self.b_save.clicked.connect(self.on_save)

        self.queue = QtGui.QListWidget(self)
        self.queue.currentItemChanged.connect(self.on_selected_changed)

        hbox = QtGui.QHBoxLayout()

        vbox = QtGui.QVBoxLayout()
        vbox.addWidget(self.b_add)
        vbox.addWidget(self.b_up)
        vbox.addWidget(self.b_down)
        vbox.addWidget(self.b_remove)
        vbox.addWidget(self.b_clear)
        vbox.addWidget(self.b_update)
        vbox.addWidget(self.b_load)
        vbox.addWidget(self.b_save)

        vbox2 = QtGui.QVBoxLayout()
        vbox2.addWidget(self.queue)
        self.stack = QtGui.QStackedWidget()
        vbox2.addWidget(self.stack)

        hbox.addWidget(self.options)
        hbox.addLayout(vbox)
        hbox.addLayout(vbox2)
        
        self.setLayout(hbox)

        self.setGeometry(800, 700, 400, 200)

    def update_plot(func):
        def wrapper(self):
            func(self)
            self.main.on_axis_changed(None)

        return wrapper
    
    @update_plot
    def on_add(self):
        if self.options.currentItem():
            name = str(self.options.currentItem().text())

            item = QtGui.QListWidgetItem(name)
            operation = Operation(name, *self.items[name])
            item.setData(QtCore.Qt.UserRole, QtCore.QVariant(operation))
            self.stack.addWidget(operation)

            self.queue.addItem(item)
            self.queue.setCurrentItem(item)

    @update_plot
    def on_up(self):
        selected_row = self.queue.currentRow()
        current = self.queue.takeItem(selected_row)
        self.queue.insertItem(selected_row - 1, current)
        self.queue.setCurrentRow(selected_row - 1)

    @update_plot
    def on_down(self):
        selected_row = self.queue.currentRow()
        current = self.queue.takeItem(selected_row)
        self.queue.insertItem(selected_row + 1, current)
        self.queue.setCurrentRow(selected_row + 1)

    @update_plot
    def on_remove(self):
        self.queue.takeItem(self.queue.currentRow())

    @update_plot
    def on_clear(self):
        self.queue.clear()

    @update_plot
    def on_update(self):
        pass

    def on_load(self):
        pass
    
    def on_save(self):
        pass

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Return:
            self.main.on_axis_changed(None)

    def on_selected_changed(self, current, previous):
        if current:
            widget = current.data(QtCore.Qt.UserRole).toPyObject()
            self.stack.addWidget(widget)
            self.stack.setCurrentWidget(widget)

    def apply_operations(self, data):
        ops = []

        for i in xrange(self.queue.count()):
            item = self.queue.item(i)
            operation = item.data(QtCore.Qt.UserRole).toPyObject()
            name = str(self.queue.item(i).text())

            data = operation.func(data, operation, linecut_type=self.main.linecut_type, linecut_coord=self.main.linecut_coord)

        return data