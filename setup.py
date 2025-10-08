"""
Setup script for MAYINI Deep Learning Framework
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="mayini-framework",
    version="0.2.0",
    author="Abhishek Adari",
    author_email="abhishekadari85@gmail.com",
    description="A comprehensive deep learning framework built from scratch in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/907-bot-collab/mayini",
    project_urls={
        "Bug Tracker": "https://github.com/907-bot-collab/mayini/issues",
        "Documentation": "https://github.com/907-bot-collab/mayini",
        "Source Code": "https://github.com/907-bot-collab/mayini",
    },
    
    # Package discovery - CRITICAL FIX
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Include all Python files
    include_package_data=True,
    
    # Dependencies
    install_requires=[
        "numpy>=1.19.0,<2.0.0",
    ],
    
    # Extra dependencies for development
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    
    # Entry points (if needed)
    entry_points={
        "console_scripts": [
            # Add any CLI commands here if needed
        ],
    },
    
    # Keywords for PyPI
    keywords="deep-learning, machine-learning, neural-networks, pytorch-like, educational, ai, tensor",
    
    # License
    license="MIT",
)
