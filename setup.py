#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io
import os
import sys
from pathlib import Path
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data
NAME = "hNMF"
DESCRIPTION = "Hierarchical NMF"
URL = "https://github.com/estasney/hNMF"
EMAIL = "estasney@users.noreply.github.com"
AUTHOR = "Eric Stasney"
REQUIRES_PYTHON = ">=3.8.0"
VERSION = "0.2.2"

REQUIRED = ["networkx>=2.3", "scikit-learn>=1.0.1", "numpy", "scipy", "rich"]

EXTRAS = {}

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
        long_description = "\n" + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

about = {"__version__": VERSION}


class BaseCommand(Command):
    def run(self) -> None:
        raise NotImplementedError

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass


class BuildCommand(BaseCommand):
    """Build packages"""

    description = "Build package"
    user_options = []

    def run(self):
        self.status("Removing Old Builds")
        try:
            rmtree(os.path.join(here, "build"))
        except OSError:
            pass
        self.status("Removing Old Distributions")
        try:
            rmtree(os.path.join(here, "dist"))
        except OSError:
            pass
        os.system("python setup.py build sdist bdist_wheel")
        self.status("Done")
        sys.exit()


class UploadCommand(BaseCommand):
    """Support setup.py upload."""

    description = "Build and publish the package."
    user_options = []

    def run(self):
        self.status("Uploading the package to PyPI via Twine…")
        import keyring

        dist_path = str(Path(here) / "dist" / "*")

        os.system(
            f"twine upload -u estasney -p {keyring.get_password('TWINE', 'estasney')} {dist_path}"
        )

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=False,
    data_files=[],
    license="MIT",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    cmdclass={"upload": UploadCommand, "package": BuildCommand},
)
