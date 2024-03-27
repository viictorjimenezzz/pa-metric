from setuptools import setup, find_packages

setup(
    name='pametric',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'typing',
        'tqdm',
        'torch',
        'torchmetrics',
        'numpy',
        'pandas',

    ],
)