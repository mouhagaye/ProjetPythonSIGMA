from setuptools import setup, find_packages
from distutils.version import LooseVersion
# import openturns as ot


# check the version of openturns
# ot_version_require = '1.12rc1'
# if LooseVersion(ot.__version__) != ot_version_require:
#     raise Exception('Version of openturns must be : {}, found {}.'.
#         format(ot_version_require, ot.__version__))
# load the version from the file
with open("VERSION", 'r') as fic:
    version = fic.read()

# set the parameter of the setup
setup(name='gradient_descent', # define the name of the package
    version=version,
    description='Module for implementing Gradient Descent',
    author='Mouhamadou Gaye',
    author_email='mouhagaye2015@gmail.com',
    # define some scripts as executable
    scripts=['src/gradient_descent.py'],
    packages=find_packages(), # namespace of the package
    # define where the package "quantile" is located
    test_suite='test', # subclass for unittest
    # some additional data included in the package
    package_data={'data': ['data/*.pkl']},
    # List of dependancies
    install_requires= ['numpy>=1.13.3',]
)
