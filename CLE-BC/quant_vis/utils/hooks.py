import torch.nn as nn


def is_model_quantizable(model: nn.Module, weight_or_act: str) -> bool:
    """
    A function for ehcking if the model has any quantizable weights and/or activations.

    Inputs:
    - model:                Our quantizable model.
    - weight_or_act (str):  Whether we are checking for quantizable weights, activations, or both.
                            Options: ['activation', 'weight', 'both'].

    Outputs:
    - return (bool): whether or not the model has a quantizable weight and/or activation.
    """

    if weight_or_act not in ["weight", "activation", "both"]:
        raise ValueError(
            "`weight_or_act` should be a string equal to `weight`, `activation`, or `both`"
        )

    # Check weights
    quantizable_model = False
    if weight_or_act in ["weight", "both"]:
        for name, _ in model.named_parameters():
            if "weight_fake_quant" in name:
                quantizable_model = True

        if not quantizable_model:
            return False

    # Check activations
    if weight_or_act in ["activation", "both"]:
        for name, module in model.named_modules():
            if hasattr(module, "activation_post_process") and hasattr(
                module.activation_post_process, "qscheme"
            ):
                quantizable_model = True

        if not quantizable_model:
            return False

    return True
