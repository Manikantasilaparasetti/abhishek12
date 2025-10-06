# Create debugged and improved versions (fixed)

# DEBUGGED SETUP.PY - Simplified version that defers to pyproject.toml
debugged_setup_py = '''#!/usr/bin/env python3
"""
MAYINI Deep Learning Framework
Minimal setup.py for backwards compatibility - main config in pyproject.toml
"""

from setuptools import setup

# Use pyproject.toml for configuration
setup()
'''

# DEBUGGED PYPROJECT.TOML - Fixed all syntax and compatibility issues
debugged_pyproject_toml = '''[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mayini-framework"
version = "0.1.2"
description = "A comprehensive deep learning framework with Tensor operations, ANN, CNN, and RNN implementations"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "Abhishek Adari", email = "abhishekadari85@gmail.com"}
]
maintainers = [
    {name = "Palivela Giridhar", email = "nanipalivela830@gmail.com"}
]
keywords = [
    "deep-learning", 
    "machine-learning", 
    "neural-networks", 
    "tensor", 
    "pytorch-like", 
    "framework",
    "autograd",
    "neural-network"
]
classifiers = [
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
    "Typing :: Typed",
]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.21.0,<2.0.0",
    "matplotlib>=3.5.0,<4.0.0",
    "seaborn>=0.11.0,<1.0.0",
    "tqdm>=4.64.0,<5.0.0",
    "scikit-learn>=1.1.0,<2.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0,<8.0",
    "pytest-cov>=4.0,<5.0",
    "black>=23.0,<25.0",
    "flake8>=5.0,<7.0",
    "mypy>=1.0,<2.0",
    "pre-commit>=3.0,<4.0"
]
docs = [
    "sphinx>=5.0,<8.0",
    "sphinx-rtd-theme>=1.0,<3.0",
    "myst-parser>=0.18,<3.0"
]
test = [
    "pytest>=7.0,<8.0",
    "pytest-cov>=4.0,<5.0",
    "pytest-xdist>=3.0,<4.0"
]
examples = [
    "jupyter>=1.0,<2.0",
    "notebook>=6.4,<8.0",
    "ipywidgets>=8.0,<9.0"
]
all = [
    "mayini-framework[dev,docs,test,examples]"
]

[project.urls]
Homepage = "https://github.com/907-bot-collab/mayini"
Documentation = "https://mayini-framework.readthedocs.io/"
Repository = "https://github.com/907-bot-collab/mayini"
"Bug Tracker" = "https://github.com/907-bot-collab/mayini/issues"
"Changelog" = "https://github.com/907-bot-collab/mayini/blob/main/CHANGELOG.md"
"Discussions" = "https://github.com/907-bot-collab/mayini/discussions"

[tool.setuptools]
package-dir = {"" = "src"}
zip-safe = false

[tool.setuptools.packages.find]
where = ["src"]
include = ["mayini*"]

[tool.setuptools.package-data]
mayini = ["py.typed"]

[tool.black]
line-length = 88
target-version = ["py38", "py39", "py310", "py311", "py312"]
include = "\\\\.pyi?$"
extend-exclude = """
/(
  # directories
  \\\\.eggs
  | \\\\.git
  | \\\\.hg
  | \\\\.mypy_cache
  | \\\\.tox
  | \\\\.venv
  | build
  | dist
)/
"""

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["mayini"]
known_third_party = ["numpy", "matplotlib", "seaborn", "sklearn", "tqdm"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "matplotlib.*",
    "seaborn.*",
    "sklearn.*",
    "tqdm.*"
]
ignore_missing_imports = true

[tool.pytest.ini_options]
minversion = "7.0"
addopts = [
    "-ra",
    "-q", 
    "--strict-markers",
    "--strict-config",
    "--cov=mayini",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml"
]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "unit: marks tests as unit tests",
    "integration: marks tests as integration tests",
    "gpu: marks tests that require GPU"
]

[tool.coverage.run]
source = ["src/mayini"]
omit = [
    "tests/*",
    "*/__pycache__/*",
    "*/site-packages/*"
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "class .*\\\\(.*Protocol.*\\\\):",
    "@(abc\\\\.)?abstractmethod"
]
show_missing = true
precision = 2

[tool.ruff]
target-version = "py38"
line-length = 88
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501",  # line too long, handled by black
    "B008",  # do not perform function calls in argument defaults
    "C901",  # too complex
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]
"tests/*" = ["B018", "B011", "B017"]
'''

print("✅ DEBUGGED SETUP.PY:")
print(f"Length: {len(debugged_setup_py)} characters")
print("- Simplified to defer to pyproject.toml")
print("- Eliminates configuration conflicts")
print()

print("✅ DEBUGGED PYPROJECT.TOML:")  
print(f"Length: {len(debugged_pyproject_toml)} characters")
print("- Fixed regex escaping issues")
print("- Added proper version bounds for dependencies")
print("- Added comprehensive tool configurations")
print("- Updated to modern Python versions (3.8+)")
print("- Added better package discovery settings")

# Store the corrected files for later use
corrected_files = {
    'setup.py': debugged_setup_py,
    'pyproject.toml': debugged_pyproject_toml
}

print(f"\n🎯 FILES READY FOR REPLACEMENT")
