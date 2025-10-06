"""
Loss functions for training neural networks.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Union
from ..tensor import Tensor
from .modules import Module


class LossFunction(Module):
    """Base class for all loss functions."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"Invalid reduction '{reduction}'. Choose from 'mean', 'sum', 'none'")
        self.reduction = reduction
    
    @abstractmethod
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute loss between predictions and targets."""
        pass
    
    def apply_reduction(self, loss: Tensor) -> Tensor:
        """Apply reduction to loss tensor."""
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # none
            return loss


class MSELoss(LossFunction):
    """Mean Squared Error Loss for regression tasks."""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = predictions - targets
        squared_diff = diff * diff
        return self.apply_reduction(squared_diff)
    
    def __repr__(self):
        return f"MSELoss(reduction={self.reduction})"


class MAELoss(LossFunction):
    """Mean Absolute Error Loss for regression tasks."""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = predictions - targets
        abs_diff_data = np.abs(diff.data)
        abs_diff = Tensor(abs_diff_data, requires_grad=(predictions.requires_grad or targets.requires_grad))
        
        # Handle gradient computation for absolute value
        if predictions.requires_grad or targets.requires_grad:
            abs_diff.op = "AbsBackward"
            abs_diff.is_leaf = False
            
            def _backward():
                grad = np.sign(diff.data) * abs_diff.grad
                
                if predictions.requires_grad:
                    if predictions.grad is None:
                        predictions.grad = grad
                    else:
                        predictions.grad = predictions.grad + grad
                
                if targets.requires_grad:
                    if targets.grad is None:
                        targets.grad = -grad
                    else:
                        targets.grad = targets.grad - grad
            
            abs_diff._backward = _backward
            abs_diff.prev = {predictions, targets}
        
        return self.apply_reduction(abs_diff)
    
    def __repr__(self):
        return f"MAELoss(reduction={self.reduction})"


class CrossEntropyLoss(LossFunction):
    """Cross-Entropy Loss for multi-class classification."""
    
    def __init__(self, reduction: str = "mean", ignore_index: int = -100):
        super().__init__(reduction)
        self.ignore_index = ignore_index
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Apply log-softmax for numerical stability
        log_probs = self._log_softmax(predictions)
        
        # Convert targets to integers if needed
        if targets.dtype != np.int64:
            targets_int = targets.data.astype(np.int64)
        else:
            targets_int = targets.data
        
        # Compute negative log-likelihood
        batch_size = predictions.shape[0]
        loss_data = np.zeros(batch_size, dtype=np.float32)
        
        for i in range(batch_size):
            target_class = targets_int[i]
            if target_class != self.ignore_index:
                loss_data[i] = -log_probs.data[i, target_class]
        
        loss = Tensor(loss_data, requires_grad=predictions.requires_grad)
        loss.op = "CrossEntropyLossBackward"
        loss.is_leaf = False
        
        def _backward():
            if predictions.requires_grad:
                grad = np.zeros_like(predictions.data)
                
                for i in range(batch_size):
                    target_class = targets_int[i]
                    if target_class != self.ignore_index:
                        # Gradient of cross-entropy w.r.t. logits
                        probs = np.exp(log_probs.data[i])
                        grad[i] = probs * loss.grad[i]
                        grad[i, target_class] -= loss.grad[i]
                
                if self.reduction == "mean":
                    # Count valid samples for averaging
                    valid_samples = np.sum(targets_int != self.ignore_index)
                    if valid_samples > 0:
                        grad = grad / valid_samples
                elif self.reduction == "sum":
                    pass  # No additional scaling needed
                
                if predictions.grad is None:
                    predictions.grad = grad
                else:
                    predictions.grad = predictions.grad + grad
        
        loss._backward = _backward
        loss.prev = {predictions}
        
        return self.apply_reduction(loss)
    
    def _log_softmax(self, x: Tensor) -> Tensor:
        """Numerically stable log-softmax."""
        x_max = np.max(x.data, axis=1, keepdims=True)
        shifted = x.data - x_max
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs_data = shifted - log_sum_exp
        
        return Tensor(log_probs_data, requires_grad=x.requires_grad)
    
    def __repr__(self):
        return f"CrossEntropyLoss(reduction={self.reduction}, ignore_index={self.ignore_index})"


class BCELoss(LossFunction):
    """Binary Cross-Entropy Loss for binary classification."""
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        # Clamp predictions to prevent log(0)
        eps = 1e-8
        predictions_clamped = np.clip(predictions.data, eps, 1 - eps)
        
        # Compute BCE: -[y*log(p) + (1-y)*log(1-p)]
        log_p = np.log(predictions_clamped)
        log_1_minus_p = np.log(1 - predictions_clamped)
        
        bce_data = -(targets.data * log_p + (1 - targets.data) * log_1_minus_p)
        bce = Tensor(bce_data, requires_grad=predictions.requires_grad)
        bce.op = "BCELossBackward"
        bce.is_leaf = False
        
        def _backward():
            if predictions.requires_grad:
                # Gradient of BCE w.r.t. predictions
                grad = -(targets.data / predictions_clamped - 
                        (1 - targets.data) / (1 - predictions_clamped)) * bce.grad
                
                if predictions.grad is None:
                    predictions.grad = grad
                else:
                    predictions.grad = predictions.grad + grad
        
        bce._backward = _backward
        bce.prev = {predictions}
        
        return self.apply_reduction(bce)
    
    def __repr__(self):
        return f"BCELoss(reduction={self.reduction})"


class HuberLoss(LossFunction):
    """Huber Loss for robust regression (less sensitive to outliers)."""
    
    def __init__(self, reduction: str = "mean", delta: float = 1.0):
        super().__init__(reduction)
        self.delta = delta
    
    def forward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        diff = predictions - targets
        abs_diff = np.abs(diff.data)
        
        # Huber loss: 0.5 * diff^2 if |diff| <= delta, else delta * (|diff| - 0.5 * delta)
        is_small = abs_diff <= self.delta
        huber_data = np.where(
            is_small,
            0.5 * diff.data ** 2,
            self.delta * (abs_diff - 0.5 * self.delta)
        )
        
        huber = Tensor(huber_data, requires_grad=(predictions.requires_grad or targets.requires_grad))
        huber.op = f"HuberLossBackward(delta={self.delta})"
        huber.is_leaf = False
        
        def _backward():
            if predictions.requires_grad or targets.requires_grad:
                # Gradient of Huber loss
                grad = np.where(
                    is_small,
                    diff.data,  # For |diff| <= delta: gradient is diff
                    self.delta * np.sign(diff.data)  # For |diff| > delta: gradient is delta * sign(diff)
                ) * huber.grad
                
                if predictions.requires_grad:
                    if predictions.grad is None:
                        predictions.grad = grad
                    else:
                        predictions.grad = predictions.grad + grad
                
                if targets.requires_grad:
                    if targets.grad is None:
                        targets.grad = -grad
                    else:
                        targets.grad = targets.grad - grad
        
        huber._backward = _backward
        huber.prev = {predictions, targets}
        
        return self.apply_reduction(huber)
    
    def __repr__(self):
        return f"HuberLoss(reduction={self.reduction}, delta={self.delta})"
