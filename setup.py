from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name="linalgeuc",
    version="0.1.0",
    description="Linear algebra library and graphing calculator",
    long_description=readme,
    author="robertgamer4",
    url="https://github.com/robertgamer4/linalgeuc",
    license=license,
    packages=find_packages(),
    python_requires='>=3.3'
)
