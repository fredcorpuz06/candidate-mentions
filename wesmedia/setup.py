from setuptools import setup

setup(
    name="wmp",
    version="0.0.1",
    description="Sample package for Wesleyan Media Project",
    author="Frederick Corpuz",
    author_email="fcorpuz@wesleyan.edu",
    packages=['wmp'],
    install_requires=[
        "pandas>=0.24.2",
        "numpy>=1.16.3"
    ])
