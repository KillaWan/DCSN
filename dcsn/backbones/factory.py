"""Construct the exact current backbones from resolved legacy backbone kwargs.

No experiment-profile defaults, normalization modules, or source-tree imports.
The wrapper nesting intentionally matches the old ``model.fm.state_dict()``.
Callers must preserve the normalization-before-backbone construction order and
seed both NumPy and Torch before construction, including checkpoint replay.
"""

from types import SimpleNamespace

from torch import nn


def _freeze_running_stats(module, args):
    if not any(parameter.requires_grad for parameter in module.parameters()):
        for child in module.modules():
            if isinstance(child, nn.modules.batchnorm._BatchNorm):
                child.eval()


class FEDformerBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        from ._fedformer import Model

        self.pred_len = int(config["pred_len"])
        self.label_len = int(config["label_len"])
        self.model = Model(SimpleNamespace(**config))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)


class AutoformerBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        from ._autoformer import Model

        self.model = Model(SimpleNamespace(**config))
        self.pred_len = config["pred_len"]
        self.label_len = config["label_len"]
        self.resolved_config = config

    def forward(self, x, x_mark, decoder, decoder_mark):
        return self.model(x, x_mark, decoder, decoder_mark)


def build_backbone(name, config):
    """Return an old-``fm`` compatible module; never reseed or mutate config.

    ``config`` is the parent's resolved old backbone kwargs, not the training
    config or the profile's sometimes-inactive ``backbone_config`` metadata.
    Unsupported upstream branches fail before constructing any parameters.
    ``FEDformer`` means the current official Fourier model.
    """
    options = dict(config)
    if name == "DLinear":
        from ._dlinear import Model

        model = Model(SimpleNamespace(**options, moving_avg=25))
    elif name in {"FEDformer", "Autoformer"}:
        if options["embed_type"] != 0 or options["embed"] != "timeF":
            raise ValueError("Current former backbones require embed_type=0 and embed='timeF'")
        if not isinstance(options["moving_avg"], int):
            raise ValueError("Current former backbones require an integer moving_avg")
        if name == "Autoformer":
            model = AutoformerBackbone(options)
        else:
            if options["version"] != "Fourier":
                raise ValueError("Only the current Fourier FEDformer is vendored")
            if options["n_heads"] != 8:
                raise ValueError("The upstream Fourier kernels require exactly eight heads")
            model = FEDformerBackbone(options)
    else:
        raise ValueError(f"Unsupported current backbone: {name!r}")
    model.register_forward_pre_hook(_freeze_running_stats)
    return model
