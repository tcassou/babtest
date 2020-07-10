# -*- coding: utf-8 -*-
from distutils.core import setup


version = '1.0.6'

setup(
    name='babtest',
    packages=['babtest', 'babtest.models'],
    version=version,
    description='Python package for Bayesian Tests / AB Testing.',
    url='https://github.com/tcassou/babtest',
    download_url='https://github.com/tcassou/babtest/archive/{}.tar.gz'.format(version),
    keywords=['python', 'bayesian', 'AB', 'test'],
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    install_requires=[
        'numpy>=1.11.3',
        'scipy>=0.18.1',
        'pymc>=2.3.6',
        'matplotlib>=1.5.3',
        'genty>=1.3.2',
        'nose>=1.3.7',
    ],
)
