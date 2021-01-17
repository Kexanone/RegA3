#!/usr/bin/env python3

from setuptools import setup
from setuptools import find_packages

with open('README.md', 'r') as stream:
    readme = stream.read()

about={}
with open('src/rega3/__version__.py', 'r') as stream:
    exec(stream.read(), about)

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    license=about['__license__'],
    author=about['__author__'],
    url=about['__url__'],
    packages=find_packages("src"),
    package_dir={'': 'src'},
    python_requires=">=3.6.*",
)
