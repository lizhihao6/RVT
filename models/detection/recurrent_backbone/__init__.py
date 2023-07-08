from omegaconf import DictConfig

from .et_rnn import RNNDetector as ETRNNDetector


def build_recurrent_backbone(backbone_cfg: DictConfig):
    name = backbone_cfg.name
    if name == 'ETRNN':
        return ETRNNDetector(backbone_cfg)
    else:
        raise NotImplementedError
