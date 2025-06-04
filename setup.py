#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="cooperative_matrix_game",
    version="0.0.1",
    author="Paresh R. Chaudhary",
    author_email="pareshrc@uw.edu",
    packages=["env", "policy", "runner", "trainer", "utils"],
    description="Toy cooperative matrix games",
    install_requires=[]
)
