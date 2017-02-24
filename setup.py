from setuptools import setup, find_packages

setup(name='qtplot',
      version='0.1.0.dev21',
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
        'vispy',
      ],
      package_data={
        '': ['*.npy']
      },
      entry_points={
        'console_scripts': ['qtplot-console = qtplot.qtplot:main'],
        'gui_scripts': ['qtplot = qtplot.qtplot:main']
      })
