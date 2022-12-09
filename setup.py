from setuptools import setup
from setuptools import find_packages

VERSION = "0.0.1"
DESCRIPTION = "Lomas is a DCN traffic demand modeling program"

setup(
    name='lomas',
    version=VERSION,
    description=DESCRIPTION,
    author="Zhiwen Liu",
    include_package_data=True,
    packages=find_packages(exclude=("tests",)),
    install_requires=[
        "pandas>=0.23.1",
    ],
)