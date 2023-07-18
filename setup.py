#!/usr/bin/env python
# Copyright (C) 2021-2022 Guillaume Jouvet <guillaume.jouvet@unil.ch>
# Published under the GNU GPL (Version 3)

from setuptools import setup, find_packages

setup(
    name="igm",
    version="2.0.0",
#    cmdclass=versioneer.get_cmdclass(),
    author="Guillaume Jouvet",
    author_email="guillaume.jouvet@unil.ch",
    description="The Instructed Glacier Model",
    url="https://github.com/jouvetg/igm",
    license="gpl-3.0",
    package_dir={"": "igm"},
)
