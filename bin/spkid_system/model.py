import sys
import os
import math
from argparse import ArgumentParser

import pytorch_lightning as pl
from pytorch_lightning.core.module import LightningModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import optim
from argparse import ArgumentParser
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging, Callback

import torchmetrics
from speechbrain.inference.speaker import EncoderClassifier

import logging

def lnorm(x):
    mx = torch.sqrt(torch.sum(x ** 2, axis=1, keepdims=True)) + 1e-10
    return math.sqrt(x.shape[1]) * x / mx


def label_reg_loss(y_pred, y_true_mean, eps=1e-10):
    #y_pred = y_pred
    y_true_mean = y_true_mean.clamp(eps, 1)  
    y_pred_mean = torch.mean(y_pred, dim=0)
    return torch.sum(y_true_mean * torch.log(y_true_mean / y_pred_mean + eps), dim=-1)


class SegmentClassificationModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.hparams.num_speakers = len(self.hparams.labels)

        # if you specify an example input, the summary will show input/output for each layer
        self.example_input_array = (torch.rand((8,  16000 * 5)), torch.ones(8).long() * 16000 * 5, torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]))

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):

        self.pretrained = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
        
        # FIXME: derive from model
        self.embed_dim = 192
   
        self.classification_layers = nn.Sequential(
            nn.Linear(self.embed_dim, self.hparams.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.hparams.dropout_prob),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(self.hparams.dropout_prob),            
            nn.Linear(self.hparams.hidden_dim, self.hparams.num_speakers)
        )

        self.valid_recall90 = torchmetrics.classification.BinaryRecallAtFixedPrecision(min_precision=0.9, thresholds=21)
        self.valid_recall90_all = torchmetrics.classification.BinaryRecallAtFixedPrecision(min_precision=0.9, thresholds=21)
        self.test_recall = torchmetrics.classification.Recall(task='binary', threshold=self.hparams.test_threshold)
        self.test_precision = torchmetrics.classification.Precision(task='binary', threshold=self.hparams.test_threshold)
        self.test_recall_at_95_precision = torchmetrics.classification.BinaryRecallAtFixedPrecision(min_precision=0.95, thresholds=21)

    def compute_embeddings(self, wavs, wav_lens):
        self.pretrained.eval()
        bs = wavs.shape[0]
        audio_len = wavs.shape[1]
        # we process 60 * 30 seconds at a time
        # otherwise, GPU might run of of memory
        # FIXME: this should be configurable of course
        max_audio_batch = 16000 * 10 * 30
        num_sub_batch = math.ceil(bs * audio_len / max_audio_batch)        
        num_items_per_sub_batch = math.ceil(bs / num_sub_batch)
        result = torch.zeros((bs, self.embed_dim), device=wavs.device)
        for i in range(num_sub_batch):
            start_item = i * num_items_per_sub_batch
            end_item = min(bs, (i + 1) * num_items_per_sub_batch)
            result[start_item:end_item] = self.compute_embeddings_single(wavs[start_item:end_item], wav_lens[start_item:end_item])
        return result
    
    def compute_embeddings_single(self, wavs, wav_lens):
        wav_lens = wav_lens / wav_lens.max()                
        feats = self.pretrained.mods.compute_features(wavs)
        feats = self.pretrained.mods.mean_var_norm(feats, wav_lens)
        embeddings = self.pretrained.mods.embedding_model(feats).squeeze(1)
        return embeddings


    def forward(self, wavs, wav_lens, spk_ids):
        #breakpoint()

        if self.global_step < self.hparams.freeze_backbone_steps:
            with torch.no_grad(): 
                embeddings = self.compute_embeddings(wavs, wav_lens)
        else:
            embeddings = self.compute_embeddings(wavs, wav_lens)
        
        # length normalization
        #embeddings = lnorm(embeddings)
        # Compute means over all segments belonging to the same automatically found speaker        
        unique_labels, labels_count = spk_ids.unique(dim=0, return_counts=True)
        
        res = torch.zeros((len(unique_labels), self.embed_dim), device=embeddings.device)        

        for i in unique_labels.sort()[0]:
            res[i] = embeddings[spk_ids==i].mean(dim=0)
        
        
        # another length norm
        #res = lnorm(res)

        return self.classification_layers(res).softmax(dim=-1).clamp(1e-10, 1)


    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """       
        # Currently we work with one wav at a time
        assert len(batch["segments"]) == 1
        assert len(batch["active_speaker_ids"]) == 1

        segments = batch["segments"][0]                
        segment_lens = batch["segment_lens"][0]                
        segment_speakers = batch["segment_speakers"][0]        
        active_speaker_ids = batch["active_speaker_ids"][0]
        num_spks = batch["num_spks"][0]
        num_total_speakers = batch["num_total_speakers"][0]
        num_active_speakers = len(active_speaker_ids)        
        label_props_for_show = torch.zeros((self.hparams.num_speakers)).to(segments.device)        
        label_props_for_show[active_speaker_ids] =  1.0 / num_spks
        # set special value for <unk> speaker with id 0
        label_props_for_show[0] = 1.0 * (num_spks - num_active_speakers) / num_spks

        #breakpoint()
        y_hat = self.forward(segments, segment_lens, segment_speakers)
        #breakpoint()
        loss_val = label_reg_loss(y_hat, label_props_for_show)

        lr  = torch.tensor(self.trainer.optimizers[0].param_groups[-1]['lr'], device=loss_val.device)
        self.log('train_loss', loss_val, prog_bar=True)
        self.log('lr', lr, prog_bar=True)
        
        return loss_val    

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        # Currently we work with one wav at a time
        assert len(batch["segments"]) == 1
        assert len(batch["active_speaker_ids"]) == 1

        segments = batch["segments"][0]                
        segment_lens = batch["segment_lens"][0]                
        segment_speakers = batch["segment_speakers"][0]        
        active_speaker_ids = batch["active_speaker_ids"][0]  
        num_total_speakers = batch["num_total_speakers"][0]
        num_spks = batch["num_spks"][0]     
        num_active_speakers = len(active_speaker_ids)    
        #breakpoint()    
        label_props_for_show = torch.zeros((self.hparams.num_speakers)).to(segments.device)        
        label_props_for_show[active_speaker_ids] =  1.0 / num_spks
        # set special value for <unk> speaker with id 0
        label_props_for_show[0] = 1.0 * (num_spks - num_active_speakers) / num_spks

        y_hat = self.forward(segments, segment_lens, segment_speakers)
        y_hat_max = y_hat.max(dim=0)[0]
        loss_val = label_reg_loss(y_hat, label_props_for_show)
        
        
        #breakpoint()
        active_speaker_ids_binarized = torch.zeros(len(y_hat_max), dtype=torch.int).to(y_hat_max.device)
        active_speaker_ids_binarized[active_speaker_ids] = 1
        # we don't count <unk>
        self.valid_recall90.update(y_hat_max[1:], active_speaker_ids_binarized[1:])
        # to also take into account speakers not covered my the model, we have to do some tricks
        # this is a bit of hack
        # we simulate that there are num_not_covered additional active speakers, who are detected with prob 0.0
        num_not_covered = num_total_speakers - len(active_speaker_ids)
        all_active_speaker_ids_binarized = torch.cat([active_speaker_ids_binarized[1:], torch.ones(num_not_covered, dtype=torch.int).to(y_hat_max.device)])
        all_y_hat_max = torch.cat([y_hat_max[1:], torch.zeros(num_not_covered).to(y_hat_max.device)])
        #breakpoint()
        self.valid_recall90_all.update(all_y_hat_max, all_active_speaker_ids_binarized)        

        self.log('val_loss', loss_val, prog_bar=True)

    def validation_epoch_end(self, outputs):
        recall, threshold = self.valid_recall90.compute()
        self.log('val_recall', recall, prog_bar=True)
        self.log('val_threshold', threshold, prog_bar=True)
        recall_all, threshold_all = self.valid_recall90_all.compute()
        self.log('val_recall_all', recall_all, prog_bar=True)
        self.log('val_threshold_all', threshold_all, prog_bar=True)


    def test_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        # Currently we work with one wav at a time
        assert len(batch["segments"]) == 1
        assert len(batch["active_speaker_ids"]) == 1

        segments = batch["segments"][0]                
        segment_lens = batch["segment_lens"][0]                
        segment_speakers = batch["segment_speakers"][0]        
        active_speaker_ids = batch["active_speaker_ids"][0]  
        num_total_speakers = batch["num_total_speakers"][0]
        num_spks = batch["num_spks"][0]     
        num_active_speakers = len(active_speaker_ids)    
        #breakpoint()    
        label_props_for_show = torch.zeros((self.hparams.num_speakers)).to(segments.device)        
        label_props_for_show[active_speaker_ids] =  1.0 / num_spks
        # set special value for <unk> speaker with id 0
        label_props_for_show[0] = 1.0 * (num_spks - num_active_speakers) / num_spks

        y_hat = self.forward(segments, segment_lens, segment_speakers)
        y_hat_max = y_hat.max(dim=0)[0]
        loss_val = label_reg_loss(y_hat, label_props_for_show)
        
        
        #breakpoint()
        active_speaker_ids_binarized = torch.zeros(len(y_hat_max), dtype=torch.int).to(y_hat_max.device)
        active_speaker_ids_binarized[active_speaker_ids] = 1
        # we don't count <unk>
        self.valid_recall90.update(y_hat_max[1:], active_speaker_ids_binarized[1:])
        # to also take into account speakers not covered my the model, we have to do some tricks
        # this is a bit of hack
        # we simulate that there are num_not_covered additional active speakers, who are detected with prob 0.0
        num_not_covered = num_total_speakers - len(active_speaker_ids)
        all_active_speaker_ids_binarized = torch.cat([active_speaker_ids_binarized[1:], torch.ones(num_not_covered, dtype=torch.int).to(y_hat_max.device)])
        all_y_hat_max = torch.cat([y_hat_max[1:], torch.zeros(num_not_covered).to(y_hat_max.device)])
        #breakpoint()
        self.test_recall.update(all_y_hat_max, all_active_speaker_ids_binarized)        
        self.test_precision.update(all_y_hat_max, all_active_speaker_ids_binarized)      


    def test_epoch_end(self, outputs):
        recall = self.test_recall.compute()        
        precision = self.test_precision.compute()
        self.log('test_threshold', self.hparams.test_threshold, prog_bar=True)
        self.log('test_recall', recall, prog_bar=True)
        self.log('test_precision', precision, prog_bar=True)


    def predict_step(self, batch, batch_idx):

        # Currently we work with one wav at a time
        assert len(batch["segments"]) == 1        
        segments = batch["segments"][0]                
        segment_lens = batch["segment_lens"][0]                
        segment_speakers = batch["segment_speakers"][0]        
        
        y_hat = self.forward(segments, segment_lens, segment_speakers)
        return y_hat

    def configure_optimizers(self):

        params = list(self.named_parameters())

        def is_backbone(n): return ('pretrained' in n) 

        grouped_parameters = [
            {"params": [p for n, p in params if is_backbone(n)], 'lr': self.hparams.learning_rate * self.hparams.backbone_lr_scale},
            {"params": [p for n, p in params if not is_backbone(n)], 'lr': self.hparams.learning_rate},
        ]

        if self.hparams.optimizer_name == "adam":
            optimizer = optim.Adam(grouped_parameters, lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name == "adamw":
            optimizer = optim.AdamW(grouped_parameters, lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name == "sgd":
            optimizer = optim.SGD(grouped_parameters, lr=self.hparams.learning_rate, momentum=0.5, weight_decay=1e-5)
        elif self.hparams.optimizer_name == "fusedlamb":
            optimizer = apex.optimizers.FusedLAMB(grouped_parameters, lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError()

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        return [optimizer], [scheduler]




    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        parser.add_argument('--hidden-dim', default=500, type=int)
        parser.add_argument('--use-ecapa', default=False, action='store_true')
        parser.add_argument("--freeze-backbone-steps", default=0, type=int)
        parser.add_argument("--backbone-lr-scale", default=0.01, type=float)
        parser.add_argument('--dropout-prob', default=0.1, type=float)

        parser.add_argument('--optimizer-name', default='sgd', type=str)
        parser.add_argument('--learning-rate', default=0.0005, type=float)

        return parser
