import argparse
import logging
import os
import pathlib
from functools import partial
from typing import List

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from music_source_separation.callbacks import get_callbacks
# from music_source_separation.callbacks.callback_evaluate import \
    # CallbackEvaluation
# from music_source_separation.callbacks.callback_save_checkpoints import \
    # CallbackCheckpoint
from music_source_separation.data.augmentors import Augmentor
from music_source_separation.data.data_modules import DataModule, Dataset
from music_source_separation.data.samplers import SegmentSampler
from music_source_separation.losses import get_loss_function
from music_source_separation.models.lightning_modules import (
    LitSourceSeparation, get_model_class)
from music_source_separation.optimizers.lr_schedulers import get_lr_lambda
from music_source_separation.utils import (StatisticsContainer, create_logging,
                                           get_pitch_shift_factor, read_yaml)


def get_dirs(workspace: str, task_name: str, filename: str, config_yaml: str, gpus: int) -> List[str]:
    r"""Get directories."""

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


def _get_modules(workspace: str, config_yaml: str) -> [SegmentSampler, Dataset]:
    r"""Get sampler, dataset."""

    configs = read_yaml(config_yaml)
    indexes_path = os.path.join(workspace, configs['train']['indexes_dict'])

    sample_rate = configs['train']['sample_rate']
    segment_seconds = configs['train']['segment_seconds']
    segment_samples = int(segment_seconds * sample_rate)
    mixaudio_dict = configs['train']['mixaudio']
    augmentation = configs['train']['augmentation']
    pitch_shift = augmentation['pitch_shift']
    ex_segment_samples = int(segment_samples * get_pitch_shift_factor(pitch_shift))

    batch_size = configs['train']['batch_size']
    steps_per_epoch = configs['train']['steps_per_epoch']
    mini_data = configs['train']['mini_data']

    # sampler
    train_sampler = SegmentSampler(
        indexes_path=indexes_path,
        segment_samples=ex_segment_samples,
        mixaudio_dict=mixaudio_dict,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )

    # augmentor
    augmentor = Augmentor(augmentation=augmentation)

    # dataset
    train_dataset = Dataset(augmentor, segment_samples)

    return train_sampler, train_dataset





def _get_callbacks(config_yaml, workspace, checkpoints_dir, statistics_path, logger, model, evaluate_device):

    configs = read_yaml(config_yaml)
    target_source_type = configs['train']['target_source_types'][0]
    test_hdf5s_dir = os.path.join(workspace, configs['evaluate']['test'])
    test_segment_seconds = configs['evaluate']['segment_seconds']
    sample_rate = configs['train']['sample_rate']
    test_segment_samples = int(test_segment_seconds * sample_rate)
    test_batch_size = configs['evaluate']['batch_size']

    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    
    # statistics container
    statistics_container = StatisticsContainer(statistics_path)

    # save checkpoint callback
    callback_checkpoints = CallbackCheckpoint(
        model=model,
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )
    
    # evaluation callback
    callback_eval_test = CallbackEvaluation(
        model=model,
        target_source_type=target_source_type,
        hdf5s_dir=test_hdf5s_dir,
        split='test',
        segment_samples=test_segment_samples,
        batch_size=test_batch_size,
        device=evaluate_device,
        evaluate_step_frequency=evaluate_step_frequency,
        logger=logger,
        statistics_container=statistics_container,
    )

    callbacks = [callback_checkpoints, callback_eval_test]
    # callbacks = []

    return callbacks

def train(args):
    r"""Train & evaluate and save checkpoints."""

    # arugments & parameters
    workspace = args.workspace
    gpus = args.gpus
    config_yaml = args.config_yaml
    filename = args.filename

    num_workers = 0
    distributed = True if gpus > 1 else False
    evaluate_device = "cuda" if gpus > 0 else "cpu"

    # Read config file.
    configs = read_yaml(config_yaml)
    task_name = configs['task_name']
    # target_source_type = configs['train']['target_source_types'][0]
    target_source_types = configs['train']['target_source_types']
    target_sources_num = len(target_source_types)
    sample_rate = configs['train']['sample_rate']
    channels = configs['train']['channels']
    model_type = configs['train']['model_type']
    loss_type = configs['train']['loss_type']

    learning_rate = float(configs['train']['learning_rate'])
    precision = configs['train']['precision']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    early_stop_steps = configs['train']['early_stop_steps']
    warm_up_steps = configs['train']['warm_up_steps']
    reduce_lr_steps = configs['train']['reduce_lr_steps']

    # paths
    checkpoints_dir, logs_dir, logger, statistics_path = get_dirs(
        workspace, task_name, filename, config_yaml, gpus
    )

    # sampler and dataset
    train_sampler, train_dataset = _get_modules(workspace, config_yaml)

    # data module
    data_module = DataModule(
        train_sampler=train_sampler,
        train_dataset=train_dataset,
        num_workers=num_workers,
        distributed=distributed,
    )

    # model
    Model = get_model_class(model_type)
    model = Model(input_channels=channels, target_sources_num=target_sources_num)

    # loss function
    loss_function = get_loss_function(loss_type)

    # callbacks
    # callbacks = get_callbacks(task_name, config_yaml, workspace, checkpoints_dir, statistics_path, logger, model, evaluate_device)
    callbacks = []

    # learning rate reduce function
    lr_lambda = partial(
        get_lr_lambda, warm_up_steps=warm_up_steps, reduce_lr_steps=reduce_lr_steps
    )

    # PL model
    '''
    pl_model = LitSourceSeparation(
        target_source_type=target_source_type,
        model=model,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_lambda=lr_lambda,
    )
    '''

    from music_source_separation.models.lightning_modules import Musdb18BatchDataPreprocessor
    batch_data_preprocessor = Musdb18BatchDataPreprocessor(target_source_types)

    pl_model = LitSourceSeparation(
        # target_source_types=target_source_types,
        batch_data_preprocessor=batch_data_preprocessor,
        model=model,
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
        plugins=[DDPPlugin(find_unused_parameters=True)],
        profiler='simple',
    )

    # Fit, evaluate, and save checkpoints.
    trainer.fit(pl_model, data_module)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    subparsers = parser.add_subparsers(dest="mode")

    parser_train = subparsers.add_parser("train")
    parser_train.add_argument("--workspace", type=str, required=True)
    parser_train.add_argument("--gpus", type=int, required=True)
    parser_train.add_argument("--config_yaml", type=str, required=True)

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    if args.mode == "train":
        train(args)

    else:
        raise Exception("Error argument!")
