# Create corrected training/__init__.py
corrected_training_init = '''"""
Training utilities for MAYINI Deep Learning Framework.
"""

from .trainer import DataLoader, Trainer, Metrics, EarlyStopping

__all__ = [
    'DataLoader', 'Trainer', 'Metrics', 'EarlyStopping'
]
'''

print("✅ CORRECTED training/__init__.py created")
print(f"Length: {len(corrected_training_init)} characters")
