# -*- coding: utf-8 -*-
from distutils.core import setup


setup(
    name='babtest',
    packages=['babtest'],
    version='1.0.0',
    description='Python package for Bayesian Tests / AB Testing.',
    url='https://github.com/tcassou/babtest',
    download_url='https://github.com/tcassou/babtest/archive/1.0.0.tar.gz',
    keywords=['python', 'bayesian', 'A/B', 'test'],
    license='MIT',
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
    ],
    install_requires=[
        'numpy==1.11.3',
        'scipy==0.18.1',
        'pymc==2.3.6',
        'matplotlib==1.5.3',
        'genty==1.3.2',
        'nose==1.3.7',
    ],
)
