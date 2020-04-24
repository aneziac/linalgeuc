from setuptools import setup, find_packages


with open('LICENSE') as f:
    license = f.read()

setup(
    name="linalgeuc",
    version="0.1.2",
    description="Geometry editor and graphing calculator in Euclidean space.",
    author="robertgamer4",
    url="https://github.com/robertgamer4/linalgeuc",
    license=license,
    packages=find_packages(),
    python_requires='>=3.3'
)
