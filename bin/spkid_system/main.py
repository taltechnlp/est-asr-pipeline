#! /usr/bin/env python3

import sys
import os
import logging
from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.core.module import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback
import data
from model import SegmentClassificationModel


seed_everything(234)

def main(args):
    if args.train_datadir is not None and args.dev_datadir is not None:
        train_dataset = data.WeaklySupervisedSpeakerDataset(args.train_datadir, extract_segments=True, **vars(args))
        dev_dataset = data.WeaklySupervisedSpeakerDataset(args.dev_datadir,
                labels=train_dataset.labels,
                extract_segments=True, segments_per_spk=0)

        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=1,
                shuffle=True,
                num_workers=8,
                drop_last=True,
                pin_memory=True)

        dev_loader = torch.utils.data.DataLoader(            
                dataset=dev_dataset,
                batch_size=1,                
                shuffle=False,                
                num_workers=4)

        if (args.load_checkpoint):
            model = SegmentClassificationModel.load_from_checkpoint(args.load_checkpoint) 
            assert model.hparams.labels == train_dataset.labels
            model.hparams.learning_rate = args.learning_rate
        else:
            model = SegmentClassificationModel(labels=train_dataset.labels, **vars(args))

        checkpoint_callback = ModelCheckpoint(
                save_top_k=4,
                save_last=True,
                verbose=True,
                monitor='val_recall',
                mode='max'                
        )    

        callbacks=[checkpoint_callback]
        trainer = Trainer.from_argparse_args(args, callbacks=callbacks)    
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader)

    elif args.test_datadir is not None and args.load_checkpoint is not None:
        model = SegmentClassificationModel.load_from_checkpoint(args.load_checkpoint, test_threshold=args.test_threshold)
        print("Loaded model")

        predict_dataset = data.WeaklySupervisedSpeakerDataset(args.test_datadir, 
                labels=model.hparams.labels,
                extract_segments=False, segments_per_spk=0)
        predict_dataset = data.WeaklySupervisedSpeakerDataset(args.test_datadir, labels=model.hparams.labels, extract_segments=True, **vars(args))

        predict_loader = torch.utils.data.DataLoader(
                dataset=predict_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=2)

        model.eval()
        args.logger = False
        trainer = Trainer.from_argparse_args(args)
        trainer.test(model, dataloaders=predict_loader)        
    elif args.predict_datadir is not None and args.load_checkpoint is not None:
        model = SegmentClassificationModel.load_from_checkpoint(args.load_checkpoint, test_threshold=args.test_threshold, strict=False)
        print("Loaded model")

        # predict_dataset = data.WeaklySupervisedSpeakerDataset(args.predict_datadir, 
        #         labels=model.hparams.labels,
        #         extract_segments=False, segments_per_spk=0)

        predict_dataset = data.WeaklySupervisedSpeakerDataset(args.predict_datadir, extract_segments=True, **vars(args))

        predict_loader = torch.utils.data.DataLoader(
                dataset=predict_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=0)

        model.eval()
        args.logger = False
        trainer = Trainer.from_argparse_args(args)
        predictions = trainer.predict(model, dataloaders=predict_loader)        
        if args.predictions_file is not None:
            with open(args.predictions_file, "w") as f:
                for i, predictions_i in enumerate(predictions):
                    spks = predict_dataset[i]['spks']
                    for j, spk in enumerate(spks):
                        pred_spk_id = predictions_i[j].argmax()
                        print(f"{spk} {predictions_i[j, pred_spk_id]} {model.hparams.labels[pred_spk_id]}", file=f)                        
            
    else:        
        raise Exception("Either --train-datadir and --dev-datadir or --test-datadir and --load-checkpoint should be specified")


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # each LightningModule defines arguments relevant to it
    parser = SegmentClassificationModel.add_model_specific_args(parent_parser, root_dir)
    parser = data.WeaklySupervisedSpeakerDataset.add_data_specific_args(parser, root_dir)
    parser = Trainer.add_argparse_args(parser)

    # data
    parser.add_argument('--train-datadir', required=False, type=str)       
    parser.add_argument('--dev-datadir', required=False, type=str)

    parser.add_argument('--load-checkpoint', required=False, type=str)        
    parser.add_argument('--test-datadir', required=False, type=str)        
    parser.add_argument('--test-threshold', default=0.5, type=float)       

    parser.add_argument('--predict-datadir', required=False, type=str)         
    parser.add_argument('--predictions-file', required=False, type=str)         

    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)
