from setuptools import setup

# try:
#     import pypandoc
#     long_description = pypandoc.convert('README.md', 'rst')
# except (IOError, ImportError, OSError):
#     long_description = ""

setup(
    name = 'kshell-utilities',
    version = '1.3.0.0',    
    description = 'Handy utilities for handling nuclear shell model calculations from KSHELL',
    # long_description = long_description,
    url = 'https://github.com/GaffaSnobb/kshell-utilities',
    author = 'Jon Kristian Dahl',
    author_email = 'jonkd@uio.no',
    packages = ['kshell_utilities', 'tests'],
    install_requires = ['numpy', 'matplotlib', 'seaborn'],

    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: MacOS',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)