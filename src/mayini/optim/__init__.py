# Create corrected optim/__init__.py
corrected_optim_init = '''"""
Optimization algorithms for MAYINI Deep Learning Framework.
"""

from .optimizers import Optimizer, SGD, Adam, AdamW, RMSprop
from .optimizers import StepLR, ExponentialLR, CosineAnnealingLR

__all__ = [
    'Optimizer',
    'SGD', 'Adam', 'AdamW', 'RMSprop',
    'StepLR', 'ExponentialLR', 'CosineAnnealingLR'
]
'''

print("✅ CORRECTED optim/__init__.py created")
print(f"Length: {len(corrected_optim_init)} characters")
