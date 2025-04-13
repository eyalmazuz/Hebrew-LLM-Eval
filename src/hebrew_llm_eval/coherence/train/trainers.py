# src/hebrew_llm_eval/coherence/train/trainers.py

"""
Custom Hugging Face Trainer implementations for the coherence task,
including weighted loss and focal loss to handle class imbalance.

Includes a factory function `get_trainer` to retrieve the appropriate Trainer class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

# Relative import for the TrainerType enum
from ...common.enums import TrainerType  # Adjust path if necessary


# --- Focal Loss Implementation ---
class FocalLoss(nn.Module):
    """
    Focal Loss implementation based on https://arxiv.org/abs/1708.02002.

    Used to give more focus to hard-to-classify examples and address class imbalance.
    """

    def __init__(self, alpha: list[float] | torch.Tensor | None = None, gamma: float = 2.0, reduction: str = "mean"):
        """
        Initializes the FocalLoss module.

        Args:
            alpha (list or tensor, optional): Weighting factor for each class.
                Can be used to counteract class imbalance. Should be a list/tensor
                of size num_classes. If None, assumes equal weight. Defaults to None.
            gamma (float, optional): Focusing parameter. Higher values give more
                focus to hard examples. Defaults to 2.0.
            reduction (str, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum'. Defaults to 'mean'.
        """
        super().__init__()
        self.gamma = gamma
        # Ensure alpha is a tensor if provided
        if alpha is not None:
            if isinstance(alpha, list):
                alpha = torch.tensor(alpha)
            assert isinstance(alpha, torch.Tensor), "'alpha' must be a list or tensor if provided"
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Calculates the Focal Loss.

        Args:
            inputs (torch.Tensor): Raw logits from the model (shape: [batch_size, num_classes]).
            targets (torch.Tensor): Ground truth labels (shape: [batch_size]).

        Returns:
            torch.Tensor: The calculated focal loss, reduced according to self.reduction.
        """
        # Compute softmax probabilities and use log_softmax for numerical stability
        log_pt = F.log_softmax(inputs, dim=1)

        # Gather the log probabilities corresponding to the true labels
        log_pt = log_pt.gather(1, targets.view(-1, 1)).view(-1)
        # Get the probabilities (pt)
        pt = log_pt.exp()

        # Calculate the core focal loss term: -(1-pt)^gamma * log(pt)
        loss = -1 * (1 - pt) ** self.gamma * log_pt

        # Apply the alpha weighting factor if provided
        if self.alpha is not None:
            # Ensure alpha is on the same device as the inputs
            alpha_t = self.alpha.to(inputs.device)[targets.data.view(-1)]
            loss = alpha_t * loss

        # Apply the specified reduction
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        # No reduction if 'none'

        return loss


# --- Weighted Loss Trainer ---
class WeightedLossTrainer(Trainer):
    """
    Custom Hugging Face Trainer that applies class weights to the
    standard CrossEntropyLoss. Useful for imbalanced datasets.
    """

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        """
        Initializes the WeightedLossTrainer.

        Args:
            *args: Positional arguments passed to the base Trainer.
            class_weights (torch.Tensor, optional): A manual rescaling weight given to
                each class. If given, has to be a Tensor of size `num_classes`.
                Defaults to None (standard CrossEntropyLoss).
            **kwargs: Keyword arguments passed to the base Trainer.
        """
        super().__init__(*args, **kwargs)
        # Store class weights (will ensure correct device in compute_loss)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss using CrossEntropyLoss with optional class weights.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model outputs must contain 'logits' key.")

        if self.class_weights is not None:
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# --- Focal Loss Trainer ---
class FocalLossTrainer(Trainer):
    """
    Custom Hugging Face Trainer that uses Focal Loss instead of
    the default CrossEntropyLoss.
    """

    def __init__(
        self,
        *args,
        focal_loss_alpha: list[float] | torch.Tensor | None = None,
        focal_loss_gamma: float = 2.0,
        **kwargs,
    ):
        """
        Initializes the FocalLossTrainer.

        Args:
            *args: Positional arguments passed to the base Trainer.
            focal_loss_alpha (list or tensor, optional): Alpha parameter (class weighting)
                for FocalLoss. Defaults to None.
            focal_loss_gamma (float, optional): Gamma parameter (focusing parameter)
                for FocalLoss. Defaults to 2.0.
            **kwargs: Keyword arguments passed to the base Trainer.
        """
        super().__init__(*args, **kwargs)
        self.focal_loss_alpha = focal_loss_alpha
        self.focal_loss_gamma = focal_loss_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Computes the loss using the FocalLoss implementation.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model outputs must contain 'logits' key.")

        loss_fct = FocalLoss(alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# --- Factory Function ---
def get_trainer(trainer_type: TrainerType) -> type[Trainer]:
    """
    Factory function to get the appropriate Trainer class based on the type enum.

    Args:
        trainer_type (TrainerType): The enum value specifying the desired trainer class.

    Returns:
        Type[Trainer]: The Trainer class itself (not an instance), suitable for instantiation.
                       (e.g., transformers.Trainer, WeightedLossTrainer, FocalLossTrainer).

    Raises:
        ValueError: If an unknown or unsupported trainer_type is provided.
    """
    match trainer_type:
        case TrainerType.DEFAULT:
            print("Using default Hugging Face Trainer.")
            return Trainer
        case TrainerType.WEIGHTED:
            print("Using custom WeightedLossTrainer.")
            return WeightedLossTrainer
        case TrainerType.FOCAL:
            print("Using custom FocalLossTrainer.")
            return FocalLossTrainer
        case _:
            # Handle unknown types gracefully
            raise ValueError(f"Unsupported trainer type provided: {trainer_type}")
