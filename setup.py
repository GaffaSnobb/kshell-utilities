from setuptools import setup

# try:
#     import pypandoc
#     long_description = pypandoc.convert('README.md', 'rst')
# except (IOError, ImportError, OSError):
#     long_description = ""

setup(
    name = 'kshell-utilities',
    version = '1.1.0.0',    
    description = 'Handy utilities for handling nuclear shell model calculations from KSHELL',
    # long_description = long_description,
    url = 'https://github.com/GaffaSnobb/kshell-utilities',
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