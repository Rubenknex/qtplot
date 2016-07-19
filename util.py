import numpy as np
from matplotlib.ticker import ScalarFormatter


def eng_format(number, significance):
    if number == 0:
        return '0'
    elif number < 0:
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
        exp = np.floor(np.log10(range / 4 / self.division))
        self.orderOfMagnitude = exp - (exp % 3)
