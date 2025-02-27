from setuptools import setup

setup(
    name = 'kshell-utilities',
    version = '1.5.1.0',    
    description = 'Handy utilities for handling nuclear shell model calculations from KSHELL',
    url = 'https://github.com/GaffaSnobb/kshell-utilities',
    author = ['Jon Kristian Dahl', 'Johannes Heines'],
    author_email = 'jonkd@uio.no',
    packages = ['kshell_utilities'],
    install_requires = ['numpy', 'matplotlib', 'seaborn', 'scipy', 'numba', 'vum'],
    package_data = {
        'kshell_utilities': ['test_files/*.txt', 'test_files/obtd_test/*.txt'],
    },

    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
