"""
Setup script for arithmetic error localization package.
"""

from setuptools import setup, find_packages

setup(
    name="arithmetic-error-localization",
    version="1.0.0",
    description="Causal localization and correction of arithmetic errors in small language models",
    author="Anonymous",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "statsmodels>=0.14.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)