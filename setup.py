from setuptools import setup

setup(
    name = 'kshell_utilities',
    version = '0.1.0',    
    description = 'Handy utilities for handling nuclear shell model calculations from KSHELL',
    url = 'https://github.com/JonKDahl/kshell_utilities',
    author = 'Jon Kristian Dahl',
    author_email = 'jonkd@uio.no',
    license = '“Commons Clause” License Condition v1.0',
    packages = ['kshell_utilities'],
    install_requires = ['numpy', 'matplotlib'],

    classifiers = [
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)