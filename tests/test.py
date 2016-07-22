import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import numpy.testing as npt

from data import DatFile, Data

equal = npt.assert_array_equal

x = np.linspace(1, 100, 100)
y = np.linspace(1, 100, 100)
xc, yc = np.meshgrid(x, y)

d_zeros = Data(xc, yc, np.zeros((100, 100)))
d_ones = Data(xc, yc, np.ones((100, 100)))
d_twos = Data(xc, yc, np.ones((100, 100)) * 2)
d_fours = Data(xc, yc, np.ones((100, 100)) * 4)

d_min_ones = Data(xc, yc, np.zeros((100, 100)) - 1)

d_slope_one_x = Data(xc, yc, xc)
d_slope_min_one_x = Data(xc, yc, -xc)

d_slope_one_y = Data(xc, yc, yc)
d_slope_min_one_y = Data(xc, yc, -yc)

d_odd_ones = Data(xc, yc, yc % 2)

d_nan = Data(xc, yc, xc * np.nan)

def test_abs():
    # |-1| == 0
    equal(Data.abs(d_min_ones).values, np.ones((100,100)))
    # |1| == 1
    equal(Data.abs(d_ones).values, np.ones((100,100)))
    # |0| == 0
    equal(Data.abs(d_zeros).values, np.zeros((100,100)))

def test_dderiv():
    kwargs = {'Theta':0,'Method':'midpoint'}
    # dderiv(0/90/180/270) == 0
    equal(Data.dderiv(d_ones, **kwargs).values, np.zeros((99, 99)))
    kwargs = {'Theta':90,'Method':'midpoint'}
    equal(Data.dderiv(d_ones, **kwargs).values, np.zeros((99, 99)))
    kwargs = {'Theta':180,'Method':'midpoint'}
    equal(Data.dderiv(d_ones, **kwargs).values, np.zeros((99, 99)))
    kwargs = {'Theta':270,'Method':'midpoint'}
    equal(Data.dderiv(d_ones, **kwargs).values, np.zeros((99, 99)))

    # dderiv(x-slope, theta=0) == 0
    kwargs = {'Theta':0,'Method':'midpoint'}
    equal(Data.dderiv(d_slope_one_x, **kwargs).values, np.ones((99, 99)))
    # Because of imprecision in numpy.pi, the cosine of .5*pi ends up being a very small value
    #kwargs = {'Theta':90,'Method':'midpoint'}
    #equal(Data.dderiv(d_slope_one_x, **kwargs).values, np.zeros((99, 99)))
    # dderiv(x-slope, theta=180) == -1
    kwargs = {'Theta':180,'Method':'midpoint'}
    equal(Data.dderiv(d_slope_one_x, **kwargs).values, -np.ones((99, 99)))

    # dderiv(x-slope, theta=0) == 0
    kwargs = {'Theta':0,'Method':'midpoint'}
    equal(Data.dderiv(d_slope_one_y, **kwargs).values, np.zeros((99, 99)))
    # Imprecision in numpy.sin(numpy.pi), almost zero
    #kwargs = {'Theta':180,'Method':'midpoint'}
    #equal(Data.dderiv(d_slope_one_y, **kwargs).values, np.zeros((99, 99)))

def test_equalize():
    pass

def test_even_odd():
    kwargs = {'Even':True}
    equal(Data.even_odd(d_odd_ones, **kwargs).values, np.ones((50,100)))
    kwargs = {'Even':False}
    equal(Data.even_odd(d_odd_ones, **kwargs).values, np.zeros((50,100)))

def test_gradmag():
    kwargs = {'Method':'midpoint'}
    equal(Data.gradmag(d_ones, **kwargs).values, np.zeros((99,99)))
    equal(Data.gradmag(d_slope_one_x, **kwargs).values, np.ones((99,99)))
    equal(Data.gradmag(d_slope_one_y, **kwargs).values, np.ones((99,99)))
    equal(Data.gradmag(d_slope_min_one_x, **kwargs).values, np.ones((99,99)))

def test_highpass():
    pass

def test_hist2d():
    pass

def test_interp_grid():
    pass

def test_log():
    kwargs = {'Subtract offset':False,'New min':0}
    # log(-1) == NaN
    equal(Data.log(d_min_ones, **kwargs).values, np.nan*np.ones((100, 100)))
    # log(0) == -inf
    equal(Data.log(d_zeros, **kwargs).values, -np.inf*np.ones((100, 100)))
    # log(1) == log(1)
    equal(Data.log(d_ones, **kwargs).values, np.log(np.ones((100, 100))))

def test_lowpass():
    pass

def test_neg():
    # -(-1) == 1
    equal(Data.neg(d_min_ones).values, np.ones((100,100)))
    # -(0) == 0
    equal(Data.neg(d_zeros).values, np.zeros((100,100)))
    # -(1) == -1
    equal(Data.neg(d_ones).values, -np.ones((100,100)))

def test_norm_columns():
    pass

def test_norm_rows():
    pass

def test_offset():
    # 0 - 1 == -1
    kwargs = {'Offset':-1}
    equal(Data.offset(d_zeros, **kwargs).values, -np.ones((100,100)))
    # 0 + 0 == 0
    kwargs = {'Offset':0}
    equal(Data.offset(d_zeros, **kwargs).values, np.zeros((100,100)))
    # 0 + 1 == 1
    kwargs = {'Offset':1}
    equal(Data.offset(d_zeros, **kwargs).values, np.ones((100,100)))

def test_offset_axes():
    pass

def test_power():
    # 1 ^ 0 == 1
    kwargs = {'Power':0}
    equal(Data.power(d_ones, **kwargs).values, np.ones((100,100)))
    # 2 ^ 2 == 4
    kwargs = {'Power':2}
    equal(Data.power(d_twos, **kwargs).values, 4 * np.ones((100,100)))
    # 4 ^ .5 == 2
    kwargs = {'Power':0.5}
    equal(Data.power(d_fours, **kwargs).values, 2 * np.ones((100,100)))
    # -1 ^ .5 == NaN
    kwargs = {'Power':0.5}
    equal(Data.power(d_min_ones, **kwargs).values, np.nan * np.ones((100,100)))
    # 4 ^ -1 == .25
    kwargs = {'Power':-1}
    equal(Data.power(d_fours, **kwargs).values, 0.25 * np.ones((100,100)))

def test_scale_axes():
    pass

def test_scale_data():
    # 0 * 1 = 0
    kwargs = {'Factor':1}
    equal(Data.scale_data(d_zeros, **kwargs).values, np.zeros((100,100)))
    # 1 * 4 = 4
    kwargs = {'Factor':4}
    equal(Data.scale_data(d_ones, **kwargs).values, 4 * np.ones((100,100)))
    # 1 * -4 = -4
    kwargs = {'Factor':-4}
    equal(Data.scale_data(d_ones, **kwargs).values, -4 * np.ones((100,100)))
    # 1 * NaN = NaN
    kwargs = {'Factor':np.nan}
    equal(Data.scale_data(d_ones, **kwargs).values, np.nan * np.ones((100,100)))
    # NaN * 5 = NaN
    kwargs = {'Factor':5}
    equal(Data.scale_data(d_nan, **kwargs).values, np.nan * np.ones((100,100)))

def test_sub_linecut():
    kwargs = {'linecut_type':'horizontal','linecut_coord':1}
    equal(Data.sub_linecut(d_ones, **kwargs).values, np.zeros((100,100)))

    kwargs = {'linecut_type':'vertical','linecut_coord':1}
    equal(Data.sub_linecut(d_ones, **kwargs).values, np.zeros((100,100)))

    kwargs = {'linecut_type':'vertical','linecut_coord':1}
    equal(Data.sub_linecut(d_odd_ones, **kwargs).values, np.zeros((100,100)))

    kwargs = {'linecut_type':'horizontal','linecut_coord':0}
    equal(Data.sub_linecut(d_odd_ones, **kwargs).values, (yc % 2) - 1)

def test_sub_plane():
    kwargs = {'X Slope':0,'Y Slope':0}
    equal(Data.sub_plane(d_ones, **kwargs).values, np.ones((100,100)))

    # How is sub slope supposed to work.
    # Center of plane at origin or center of dataset?

    #kwargs = {'X Slope':1,'Y Slope':0}
    #equal(Data.sub_plane(d_zeros, **kwargs).values, xc)

def test_xderiv():
    equal(Data.xderiv(d_zeros, Method='midpoint').values, np.zeros((100, 99)))
    equal(Data.xderiv(d_ones, Method='midpoint').values, np.zeros((100, 99)))
    equal(Data.xderiv(d_zeros, Method='midpoint').values, np.zeros((100, 99)))
    equal(Data.xderiv(d_slope_one_x, Method='midpoint').values, np.ones((100, 99)))
    equal(Data.xderiv(d_slope_min_one_x, Method='midpoint').values, -np.ones((100, 99)))

def test_yderiv():
    equal(Data.yderiv(d_zeros, Method='midpoint').values, np.zeros((99, 100)))
    equal(Data.yderiv(d_ones, Method='midpoint').values, np.zeros((99, 100)))
    equal(Data.yderiv(d_zeros, Method='midpoint').values, np.zeros((99, 100)))
    equal(Data.yderiv(Data(xc, yc, np.copy(yc)), Method='midpoint').values, np.ones((99, 100)))
    equal(Data.yderiv(Data(xc, yc, -np.copy(yc)), Method='midpoint').values, -np.ones((99, 100)))

if __name__ == '__main__':
    test_abs()
    test_dderiv()
    test_equalize()
    test_even_odd()
    test_gradmag()
    test_highpass()
    test_hist2d()
    test_interp_grid()
    test_log()
    test_lowpass()
    test_neg()
    test_norm_columns()
    test_norm_rows()
    test_offset()
    test_offset_axes()
    test_power()
    test_scale_axes()
    test_scale_data()
    test_sub_linecut()
    test_sub_plane()
    test_xderiv()
    test_yderiv()