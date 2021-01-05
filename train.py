from __future__ import absolute_import
import train_opts as opts
import argparse
import glob
import torch
import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger()
from onmt_modules.onmt.inputters.fields import build_dynamic_fields
from onmt_modules.onmt.transforms import get_specials, get_transforms_cls
from onmt_modules.onmt.opts import train_opts
from onmt_modules.onmt.model_builder import build_base_model
from onmt_modules.onmt.utils.optimizers import Optimizer
from onmt_modules.onmt.trainer import build_trainer
from onmt_modules.onmt.train_single import _build_train_iter, _build_valid_iter
from onmt_modules.onmt.inputters.inputter import IterOnDevice

from onmt_modules.onmt.utils.parse import ArgumentParser
parser = ArgumentParser(description='train.py')
train_opts(parser)
opt, unknown = parser.parse_known_args()



def init_logger(log_file=None, log_file_level=logging.NOTSET, rotate=False):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
    

def get_fields(opt):
    ArgumentParser.validate_train_opts(opt)
    ArgumentParser.update_model_opts(opt)
    ArgumentParser.validate_model_opts(opt)
    ArgumentParser.validate_prepare_opts(opt)
    transforms_cls = get_transforms_cls(opt._all_transform)
    specials = get_specials(opt, transforms_cls)

    fields = build_dynamic_fields(
        opt, src_specials=specials['src'], tgt_specials=specials['tgt'])
    return fields, transforms_cls

def use_gpu(opt):
    """
    Creates a boolean if gpu used
    """
    return (hasattr(opt, 'gpu_ranks') and len(opt.gpu_ranks) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)

def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        if (opt.tensorboard_log_dir == model_opt.tensorboard_log_dir and
                hasattr(model_opt, 'tensorboard_log_dir_dated')):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            opt.tensorboard_log_dir_dated = model_opt.tensorboard_log_dir_dated
    else:
        model_opt = opt
    return model_opt

def model_constructor(opt, fields, transforms_cls, checkpoint, device_id):
    init_logger(opt.log_file)
    model_opt = _get_model_opts(opt, checkpoint=checkpoint)
    model = build_base_model(model_opt, fields, use_gpu(opt), checkpoint)
    model.count_parameters(log=logger.info) 
    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)
    trainer = build_trainer(
        opt, device_id, model, fields, optim)
    _train_iter = _build_train_iter(opt, fields, transforms_cls)
    train_iter = IterOnDevice(_train_iter, device_id)
    valid_iter = _build_valid_iter(opt, fields, transforms_cls)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, device_id)
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
    

def main():
    if opt.train_from:
        print('Loading checkpoint from %s' % opt.train_from)
        checkpoint = torch.load(opt.train_from,
                                map_location=lambda storage, loc: storage)
        model_opt = checkpoint['opt']
        # I don't like reassigning attributes of opt: it's not clear.
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        checkpoint = None
        model_opt = opt
        
    fields, transforms_cls = get_fields(opt)
    
    model = model_constructor(opt, fields, transforms_cls, checkpoint, device_id=-1)



if __name__ == "__main__":
    main()