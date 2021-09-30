from setuptools import setup

setup(
    name = 'kshell_utilities',
    version = '1.0.1.1',    
    description = 'Handy utilities for handling nuclear shell model calculations from KSHELL',
    url = 'https://github.com/GaffaSnobb/kshell_utilities',
    author = 'Jon Kristian Dahl',
    author_email = 'jonkd@uio.no',
    packages = ['kshell_utilities', 'tests'],
    install_requires = ['numpy', 'matplotlib'],

    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)