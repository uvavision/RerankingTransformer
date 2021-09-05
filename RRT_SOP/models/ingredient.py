from sacred import Ingredient

from .architectures import __all__, __dict__

model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    arch = 'superpoint'
    freeze_backbone = True
    num_local_features = 256
    ert_dim_feedforward=1024
    ert_nhead=4
    ert_num_encoder_layers=6
    ert_dropout=0.0 
    ert_activation="relu"
    ert_normalize_before=False


@model_ingredient.capture
def get_model(arch, num_local_features, freeze_backbone, ert_dim_feedforward, ert_nhead, ert_num_encoder_layers, ert_dropout, ert_activation, ert_normalize_before):
    keys = list(map(lambda x: x.lower(), __all__))
    index = keys.index(arch.lower())
    arch = __all__[index]
    return __dict__[arch](num_local_features=num_local_features, freeze_backbone=freeze_backbone, 
                        ert_dim_feedforward=ert_dim_feedforward, ert_nhead=ert_nhead, ert_num_encoder_layers=ert_num_encoder_layers, 
                        ert_dropout=ert_dropout, ert_activation=ert_activation, ert_normalize_before=ert_normalize_before)
