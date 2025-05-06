def get_model(name, config):
    if name == "resnet18":
        from .resnet18 import build_model
    elif name == "mobilevit":
        from .mobilevit import build_model
    elif name == "tinyvit":
        from .tinyvit import build_model
    elif name == "deit":
        from .deit import build_model
    elif name == "efficientnet_lite":
        from .efficientnet_lite import build_model
    else:
        raise ValueError(f"Unknown model: {name}")
    return build_model(config)
