from setuptools import setup, find_packages


with open('LICENSE') as f:
    license = f.read()

setup(
    name="linalgeuc",
    version="0.4.0",
    description="Rasterization engine for 3D shapes",
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
