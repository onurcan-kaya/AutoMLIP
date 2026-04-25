from setuptools import setup, find_packages

setup(
    name="automlip",
    version="0.2.0",
    description="Automated training of machine learning interatomic potentials",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "ase",
        "pyyaml",
    ],
    extras_require={
        "gap": ["quippy-ase"],
        "mace": ["mace-torch"],
        "nequip": ["nequip"],
    },
)
