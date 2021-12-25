import argparse
import logging
import os
import pathlib
from functools import partial
from typing import Dict, List, NoReturn

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torch

from bytesep.callbacks import get_callbacks
from bytesep.data.augmentors import Augmentor
from bytesep.data.batch_data_preprocessors import get_batch_data_preprocessor_class
from bytesep.data.data_modules import DataModule, Dataset
from bytesep.data.samplers import SegmentSampler
from bytesep.losses import get_loss_function
from bytesep.models.lightning_modules import LitSourceSeparation, get_model_class
from bytesep.optimizers.lr_schedulers import get_lr_lambda
from bytesep.utils import (
    check_configs_gramma,
    create_logging,
    get_pitch_shift_factor,
    read_yaml,
)


def get_dirs(
    workspace: str,
    task_name: str,
    filename: str,
    config_yaml: str,
    gpus: int,
) -> List[str]:
    r"""Get directory paths.

    Args:
        workspace: str
        task_name, str, e.g., 'musdb18'
        filenmae: str
        config_yaml: str
        gpus: int, e.g., 0 for cpu and 8 for training with 8 gpu cards

    Returns:
        checkpoints_dir: str
        logs_dir: str
        logger: pl.loggers.TensorBoardLogger
        statistics_path: str
    """

    # save checkpoints dir
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        task_name,
        filename,
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # logs dir
    logs_dir = os.path.join(
        workspace,
        "logs",
        task_name,
        filename,
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # loggings
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # tensorboard logs dir
    tb_logs_dir = os.path.join(workspace, "tensorboard_logs")
    os.makedirs(tb_logs_dir, exist_ok=True)

    experiment_name = os.path.join(task_name, filename, pathlib.Path(config_yaml).stem)
    logger = pl.loggers.TensorBoardLogger(save_dir=tb_logs_dir, name=experiment_name)

    # statistics path
    statistics_path = os.path.join(
        workspace,
        "statistics",
        task_name,
        filename,
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, logger, statistics_path


def get_data_module(
    workspace: str,
    config_yaml: str,
    num_workers: int,
    distributed: bool,
) -> DataModule:
    r"""Create data_module. Here is an example to fetch a mini-batch:

    code-block:: python

        data_module.setup()
        for batch_data_dict in data_module.train_dataloader():
            print(batch_data_dict.keys())
            break

    Args:
        workspace: str
        config_yaml: str
        num_workers: int, e.g., 0 for non-parallel and 8 for using cpu cores
            for preparing data in parallel
        distributed: bool

    Returns:
        data_module: DataModule
    """
    configs = read_yaml(config_yaml)
    input_source_types = configs['train']['input_source_types']
    target_source_types = configs['train']['target_source_types']
    paired_input_target_data = configs['train']['paired_input_target_data']
    indexes_dict_path = os.path.join(workspace, configs['train']['indexes_dict_path'])
    sample_rate = configs['train']['sample_rate']
    input_channels = configs['train']['input_channels']
    segment_seconds = configs['train']['segment_seconds']
    augmentations = configs['train']['augmentations']
    batch_size = configs['train']['batch_size']
    steps_per_epoch = configs['train']['steps_per_epoch']

    segment_samples = int(segment_seconds * sample_rate)

    if paired_input_target_data:
        assert (
            augmentations['remixing_sources'] is False
        ), "Must set remixing_sources to False if input and target data are paired."

    ex_segment_samples = get_pitch_shifted_segment_samples(
        segment_samples=segment_samples,
        augmentations=augmentations,
    )

    # sampler
    train_sampler = SegmentSampler(
        indexes_dict_path=indexes_dict_path,
        input_source_types=input_source_types,
        target_source_types=target_source_types,
        segment_samples=ex_segment_samples,
        remixing_sources=augmentations['remixing_sources'],
        mixaudio_dict=augmentations['mixaudio'],
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    # augmentor
    augmentor = Augmentor(augmentations=augmentations)

    # dataset
    train_dataset = Dataset(
        input_source_types=input_source_types,
        target_source_types=target_source_types,
        paired_input_target_data=paired_input_target_data,
        input_channels=input_channels,
        augmentor=augmentor,
        segment_samples=segment_samples,
    )

    # data module
    data_module = DataModule(
        train_sampler=train_sampler,
        train_dataset=train_dataset,
        num_workers=num_workers,
        distributed=distributed,
    )

    return data_module


def get_pitch_shifted_segment_samples(segment_samples: int, augmentations: Dict) -> int:
    r"""Get new segment samples depending on maximum pitch shift.

    Args:
        segment_samples: int
        augmentations: Dict

    Returns:
        ex_segment_samples: int
    """

    if 'pitch_shift' not in augmentations.keys():
        return segment_samples

    else:
        pitch_shift_dict = augmentations['pitch_shift']
        source_types = pitch_shift_dict.keys()

    max_pitch_shift = max(
        [pitch_shift_dict[source_type] for source_type in source_types]
    )

    ex_segment_samples = int(segment_samples * get_pitch_shift_factor(max_pitch_shift))

    return ex_segment_samples


def train(args) -> NoReturn:
    r"""Train & evaluate and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int
        config_yaml: str, path of config file for training
    """

    # arugments & parameters
    workspace = args.workspace
    gpus = args.gpus
    config_yaml = args.config_yaml
    filename = args.filename

    num_workers = 8
    distributed = True if gpus > 1 else False
    evaluate_device = "cuda" if gpus > 0 else "cpu"

    # Read config file.
    configs = read_yaml(config_yaml)
    check_configs_gramma(configs)
    task_name = configs['task_name']
    input_source_types = configs['train']['input_source_types']
    target_source_types = configs['train']['target_source_types']
    input_channels = configs['train']['input_channels']
    output_channels = configs['train']['output_channels']
    batch_data_preprocessor_type = configs['train']['batch_data_preprocessor']
    model_type = configs['train']['model_type']
    loss_type = configs['train']['loss_type']
    optimizer_type = configs['train']['optimizer_type']
    learning_rate = float(configs['train']['learning_rate'])
    precision = configs['train']['precision']
    early_stop_steps = configs['train']['early_stop_steps']
    warm_up_steps = configs['train']['warm_up_steps']
    reduce_lr_steps = configs['train']['reduce_lr_steps']
    resume_checkpoint_path = configs['train']['resume_checkpoint_path']

    target_sources_num = len(target_source_types)

    # paths
    checkpoints_dir, logs_dir, logger, statistics_path = get_dirs(
        workspace, task_name, filename, config_yaml, gpus
    )

    # training data module
    data_module = get_data_module(
        workspace=workspace,
        config_yaml=config_yaml,
        num_workers=num_workers,
        distributed=distributed,
    )

    # batch data preprocessor
    BatchDataPreprocessor = get_batch_data_preprocessor_class(
        batch_data_preprocessor_type=batch_data_preprocessor_type
    )

    batch_data_preprocessor = BatchDataPreprocessor(
        input_source_types=input_source_types, target_source_types=target_source_types
    )

    # model
    Model = get_model_class(model_type=model_type)
    model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        target_sources_num=target_sources_num,
    )

    if resume_checkpoint_path:
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        logging.info(
            "Load pretrained checkpoint from {}".format(resume_checkpoint_path)
        )

    # loss function
    loss_function = get_loss_function(loss_type=loss_type)

    # callbacks
    callbacks = get_callbacks(
        task_name=task_name,
        config_yaml=config_yaml,
        workspace=workspace,
        checkpoints_dir=checkpoints_dir,
        statistics_path=statistics_path,
        logger=logger,
        model=model,
        evaluate_device=evaluate_device,
    )
    # callbacks = []

    # learning rate reduce function
    lr_lambda = partial(
        get_lr_lambda, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps
    )

    # pytorch-lightning model
    pl_model = LitSourceSeparation(
        batch_data_preprocessor=batch_data_preprocessor,
        model=model,
        optimizer_type=optimizer_type,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_lambda=lr_lambda,
    )

    # trainer
    trainer = pl.Trainer(
        checkpoint_callback=False,
        gpus=gpus,
        callbacks=callbacks,
        max_steps=early_stop_steps,
        accelerator="ddp",
        sync_batchnorm=True,
        precision=precision,
        replace_sampler_ddp=False,
        plugins=[DDPPlugin(find_unused_parameters=False)],
        profiler='simple',
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(pl_model, data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser_train.add_argument("--gpus", type=int, required=True)
    parser_train.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    if args.mode == "train":
        train(args)

    else:
        raise Exception("Error argument!")
