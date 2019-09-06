from setuptools import setup, find_packages

setup(name='aegis_scripts',
  version='0.1.0',
  install_requires = [
    'aegis @ git+https://github.com/tehzevo/aegis-core@master#egg=aegis',
    'ml_utils @ git+https://github.com/tehzevo/ml-utils@master#egg=ml_utils',
    'tensorflow',
    'matplotlib',
    'gym',
  ],
  packages=find_packages())
