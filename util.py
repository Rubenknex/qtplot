import numpy as np
from matplotlib.ticker import ScalarFormatter

def create_kernel(x_dev, y_dev, cutoff, distr):
    distributions = {
        'gaussian': lambda r: np.exp(-(r**2) / 2.0),
        'exponential': lambda r: np.exp(-abs(r) * np.sqrt(2.0)),
        'lorentzian': lambda r: 1.0 / (r**2+1.0),
        'thermal': lambda r: np.exp(r) / (1 * (1+np.exp(r))**2)
    }
    func = distributions[distr]

    hx = np.floor((x_dev * cutoff) / 2.0)
    hy = np.floor((y_dev * cutoff) / 2.0)

    x = np.linspace(-hx, hx, hx * 2 + 1) / x_dev
    y = np.linspace(-hy, hy, hy * 2 + 1) / y_dev

    if x.size == 1: x = np.zeros(1)
    if y.size == 1: y = np.zeros(1)
    
    xv, yv = np.meshgrid(x, y)

    kernel = func(np.sqrt(xv**2+yv**2))
    kernel /= np.sum(kernel)

    return kernel

def eng_format(number, significance):
    if number < 0:
        sign = '-'
    else:
        sign = ''

    exp = int(np.floor(np.log10(abs(number))))
    exp3 = exp - (exp % 3)

    x3 = abs(number) / (10**exp3)
    
    if exp3 == 0:
        exp3_text = ''
    else:
        exp3_text = 'e%s' % exp3

    format = '%.' + str(significance) + 'f'

    return ('%s' + format + '%s') % (sign, x3, exp3_text)

class FixedOrderFormatter(ScalarFormatter):
    """Format numbers using engineering notation."""
    def __init__(self, format='%.0f', division=1e0):
        ScalarFormatter.__init__(self, useOffset=None, useMathText=None)
        #self.format = '%.' + str(significance) + 'f'
        #print format
        self.format = format
        self.division = division

    def __call__(self, x, pos=None):
        if x == 0:
            return '0'

        exp = self.orderOfMagnitude

        return self.format % ((x / self.division) / (10 ** exp))

    def _set_format(self, vmin, vmax):
        pass

    def _set_orderOfMagnitude(self, range):
        exp = np.floor(np.log10(range / 2 / self.division))
        self.orderOfMagnitude = exp - (exp % 3)