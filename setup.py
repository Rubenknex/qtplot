from setuptools import setup, find_packages

version = '0.1.2'

setup(name='qtplot',
      version=version,
      description='Data plotting and analysis tool',
      url='https://github.com/Rubenknex/qtplot',
      author='Ruben van Gulik',
      author_email='rubenvangulik@gmail.com',
      license='MIT',
      packages=['qtplot',
                'tests',
                'qtplot/colormaps',
                'qtplot/colormaps/nanoscope',
                'qtplot/colormaps/transform',
                'qtplot/colormaps/wsxm'],
      install_requires=[
        'QtPy',
        'pyopengl',
        'vispy==0.4.0',
      ],
      package_data={
        '': ['*.npy']
      },
      entry_points={
        'console_scripts': ['qtplot-console-%s = qtplot.qtplot:main' % version],
        'gui_scripts': ['qtplot-%s = qtplot.qtplot:main' % version]
      })
