import math
import os
from typing import Tuple

from omegaconf import DictConfig, open_dict

from data.utils.spatial import get_dataloading_hw


def dynamically_modify_train_config(config: DictConfig):
    with open_dict(config):
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        if slurm_job_id and slurm_job_id != '':
            config.slurm_job_id = int(slurm_job_id)

        dataset_cfg = config.dataset

        dataset_name = dataset_cfg.name
        assert dataset_name in {'gen1', 'gen4'}
        dataset_hw = get_dataloading_hw(dataset_config=dataset_cfg)

        mdl_cfg = config.model
        mdl_cfg.dataset_hw = dataset_hw
        mdl_name = mdl_cfg.name
        if mdl_name == 'rnndet':
            backbone_cfg = mdl_cfg.backbone
            backbone_name = backbone_cfg.name
            if backbone_name == 'ETRNN':
                backbone_cfg.in_res_hw = dataset_hw
            else:
                print(f'{backbone_name=} not available')
                raise NotImplementedError
            num_classes = 2 if dataset_name == 'gen1' else 3
            mdl_cfg.head.num_classes = num_classes
            print(f'Set {num_classes=} for detection head')
        else:
            print(f'{mdl_name=} not available')
            raise NotImplementedError


def _get_modified_hw_multiple_of(hw: Tuple[int, int],
                                 multiple_of: int) -> Tuple[int, ...]:
    assert isinstance(hw, tuple), f'{type(hw)=}, {hw=}'
    assert len(hw) == 2
    assert isinstance(multiple_of, int)
    assert multiple_of >= 1
    if multiple_of == 1:
        return hw
    new_hw = tuple(math.ceil(x / multiple_of) * multiple_of for x in hw)
    return new_hw
