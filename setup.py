from setuptools import setup, find_packages

setup(name='qtplot',
      version='0.1.0',
      description='',
      url='https://github.com/Rubenknex/qtplot',
      author='Ruben van Gulik',
      author_email='rubenvangulik@gmail.com',
      license='MIT',
      packages=['qtplot',
                'qtplot/colormaps',
                'qtplot/colormaps/nanoscope',
                'qtplot/colormaps/transform',
                'qtplot/colormaps/wsxm'],
      install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
        'pyopengl',
        'vispy',
      ],
      package_data={
        '': ['*.npy'],
      })
