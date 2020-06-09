from setuptools import setup, find_packages


with open('LICENSE') as f:
    license = f.read()

setup(
    name="linalgeuc",
    version="0.2.1",
    description="Geometry editor and graphing calculator in Euclidean space",
    author="aneziac",
    url="https://github.com/aneziac/linalgeuc",
    license=license,
    packages=find_packages(),
    install_requires=[
        'pygame',
        'numpy'
    ],
    python_requires='>=3.3'
)
