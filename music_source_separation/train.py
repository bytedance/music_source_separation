import argparse
import logging
import os
import pathlib
from functools import partial

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

from music_source_separation.utils import read_yaml, create_logging, StatisticsContainer
from music_source_separation.data.augmentors import Augmentor
from music_source_separation.data.data_modules import DataModule
from music_source_separation.data.samplers import SegmentSampler
from music_source_separation.models.lightning_modules import get_model_class, LitSourceSeparation
from music_source_separation.losses import get_loss_function
from music_source_separation.optimizers.lr_schedulers import get_lr_lambda
from music_source_separation.callbacks.callback_save_checkpoints import CallbackCheckpoint
from music_source_separation.callbacks.callback_evaluate import CallbackEvaluation


def get_dirs(workspace, filename, config_yaml, gpus):

    # save checkpoints dir
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # logs dir
    logs_dir = os.path.join(
        workspace, "logs", filename, 
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus)
    )
    os.makedirs(logs_dir, exist_ok=True)

    # loggings
    create_logging(logs_dir, filemode='w')
    logging.info(args)

    # tensorboard logs dir
    tb_logs_dir = os.path.join(workspace, "tensorboard_logs")
    os.makedirs(tb_logs_dir, exist_ok=True)

    experiment_name = os.path.join(filename, pathlib.Path(config_yaml).stem)
    logger = pl.loggers.TensorBoardLogger(save_dir=tb_logs_dir, name=experiment_name)

    # statistics path
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "config={},gpus={}".format(pathlib.Path(config_yaml).stem, gpus),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, logger, statistics_path


def train(args):
    r"""Train & evaluate and save checkpoints.
    """

    # arugments & parameters
    workspace = args.workspace
    gpus = args.gpus
    config_yaml = args.config_yaml
    filename = args.filename

    num_workers = 8
    distributed = True if gpus > 1 else False
    evaluate_device = "cuda" if gpus > 1 else "cpu"

    # Read config file.
    configs = read_yaml(config_yaml)
    target_source_type = configs['train']['target_source_types'][0]
    indexes_path = os.path.join(workspace, configs['train']['indexes_dict'])
    sample_rate = configs['train']['sample_rate']
    channels = configs['train']['channels']
    segment_seconds = configs['train']['segment_seconds']
    model_type = configs['train']['model_type']
    loss_type = configs['train']['loss_type']

    # if 'max_random_shift' in configs['train']:
    #     max_random_shift = configs['train']['max_random_shift']
    # else:
    #     max_random_shift = None
    max_random_shift = None

    # if 'sampler_type' in configs['train']:
    #     Sampler = eval(configs['train']['sampler_type'])

    # else:
    #     Sampler = SegmentSampler
    Sampler = SegmentSampler

    mixaudio_dict =  configs['train']['augmentation']['mixaudio']

    random_scale_dict = configs['train']['augmentation']['random_scale'] if \
        'random_scale' in configs['train']['augmentation'] else None

    # if 'random_scale' in configs['train']['augmentation']:
    #     random_scale_dict = configs['train']['augmentation']['random_scale']
    # else:
    #     random_scale_dict = None

    # if 'return_output_dict' in configs['train']:
    #     return_output_dict = True
    # else:
    #     return_output_dict = False

    learning_rate = float(configs['train']['learning_rate'])
    batch_size = configs['train']['batch_size']
    precision = configs['train']['precision']
    steps_per_epoch = configs['train']['steps_per_epoch']
    evaluate_step_frequency = configs['train']['evaluate_step_frequency']
    save_step_frequency = configs['train']['save_step_frequency']
    early_stop_steps = configs['train']['early_stop_steps']
    warm_up_steps = configs['train']['warm_up_steps']
    reduce_lr_steps = configs['train']['reduce_lr_steps']

    test_hdf5s_dir = os.path.join(workspace, configs['evaluate']['test'])
    test_segment_seconds = configs['evaluate']['segment_seconds']
    test_batch_size = configs['evaluate']['batch_size']

    segment_samples = int(segment_seconds * sample_rate)
    test_segment_samples = int(test_segment_seconds * sample_rate)
    
    # paths
    _many_dirs = get_dirs(workspace, filename, config_yaml, gpus)
    checkpoints_dir, logs_dir, logger, statistics_path = _many_dirs

    # statistics container
    statistics_container = StatisticsContainer(statistics_path)

    # augmentor
    augmentor = Augmentor(random_scale_dict)

    # data module
    data_module = DataModule(
        indexes_path=indexes_path,
        max_random_shift=max_random_shift,
        mixaudio_dict=mixaudio_dict,
        augmentor=augmentor,
        Sampler=Sampler,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        num_workers=num_workers,
        distributed=distributed,
    )

    # data_module.setup()
    # for e in data_module.train_dataloader():
    #     from IPython import embed; embed(using=False); os._exit(0)
    #     import soundfile
    #     soundfile.write(file='_zz.wav', data=e['accompaniment'][2].data.cpu().numpy().T, samplerate=44100)
    #     import librosa
    #     audio, fs = librosa.load('_zz.wav', sr=None, mono=False)

    # model
    Model = get_model_class(model_type)
    model = Model(channels)

    # if resume_checkpoint_path:
    #     logging.info('Load resume model from {}'.format(resume_checkpoint_path))
    #     resume_checkpoint = torch.load(resume_checkpoint_path)
    #     model.load_state_dict(resume_checkpoint['model'])
    #     resume_iteration = resume_checkpoint['iteration']
    #     statistics_container.load_state_dict(resume_iteration)
    #     iteration = resume_iteration
    # else:
    #     resume_iteration = 0
    #     iteration = 0

    # loss function
    loss_function = get_loss_function(loss_type)

    # callbacks
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

    # learning rate reduce function
    lr_lambda = partial(
        get_lr_lambda, 
        warm_up_steps=warm_up_steps, 
        reduce_lr_steps=reduce_lr_steps
    )

    # PL model
    pl_model = LitSourceSeparation(
        target_source_type=target_source_type,
        model=model,
        loss_function=loss_function,
        learning_rate=learning_rate,
        lr_lambda=lr_lambda
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
