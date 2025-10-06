#!/usr/bin/env python3
"""
MAYINI Deep Learning Framework
A comprehensive deep learning framework with Tensor operations, ANN, CNN, and RNN implementations.
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    """Read README.md file for long description."""
    here = os.path.abspath(os.path.dirname(__file__))
    readme_path = os.path.join(here, 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    """Read requirements.txt file."""
    here = os.path.abspath(os.path.dirname(__file__))
    requirements_path = os.path.join(here, 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'numpy>=1.21.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'tqdm>=4.64.0',
        'scikit-learn>=1.1.0'
    ]

setup(
    name="mayini-framework",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive deep learning framework with Tensor operations, ANN, CNN, and RNN implementations",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mayini-framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.17",
        ],
        "examples": [
            "tensorflow>=2.8.0",
            "torch>=1.11.0",
            "jupyter>=1.0",
            "notebook>=6.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "mayini-info=mayini.cli:info",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/mayini-framework/issues",
        "Source": "https://github.com/yourusername/mayini-framework",
        "Documentation": "https://mayini-framework.readthedocs.io/",
    },
    keywords="deep-learning machine-learning neural-networks tensor pytorch-like framework",
    include_package_data=True,
    zip_safe=False,
)
