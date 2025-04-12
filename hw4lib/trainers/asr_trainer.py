import logging
from hw4lib.trainers.base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from hw4lib.decoding.sequence_generator import SequenceGenerator
from hw4lib.utils import create_scheduler, create_optimizer
from hw4lib.model import DecoderOnlyTransformer
import torchaudio.functional as aF
import json
import torchmetrics.text as tmt
from torch.utils.data import Subset
import pandas as pd

# Initialize a logger (configure it as needed in your application)
#logger = logging.getLogger(_name_)


class ASRTrainer(BaseTrainer):
    """
    ASR (Automatic Speech Recognition) Trainer class that handles training, validation, and recognition loops.

    This trainer implements:
      1. Training loop with gradient accumulation, mixed precision training, and optional CTC loss.
      2. Validation loop for model evaluation.
      3. Recognition capabilities with different decoding strategies (greedy, beam search).
      4. Language model shallow fusion during recognition.
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)

        # Initialize CrossEntropyLoss with padding index and label smoothing.
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss'].get('label_smoothing', 0.0)
        )

        # Initialize CTC loss if ctc_weight > 0.
        self.ctc_criterion = None
        self.ctc_weight = self.config['loss'].get('ctc_weight', 0.0)
        if self.ctc_weight > 0:
            self.ctc_criterion = nn.CTCLoss(
                blank=self.tokenizer.pad_id,
                zero_infinity=True
            )

    def _train_epoch(self, dataloader):
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data.
        Returns:
            Tuple containing training metrics and the latest attention weights.
        """
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False,
                         position=0, desc="[Training ASR]")
        running_ce_loss = 0.0
        running_ctc_loss = 0.0
        running_joint_loss = 0.0
        total_tokens = 0
        running_att = None

        # Zero the optimizer at the start of a new gradient accumulation cycle.
        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths = batch
            feats = feats.to(self.device)
            targets_shifted = targets_shifted.to(self.device)
            targets_golden = targets_golden.to(self.device)
            feat_lengths = feat_lengths.to(self.device)
            transcript_lengths = transcript_lengths.to(self.device)

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                model_output = self.model(feats, targets_shifted, feat_lengths, transcript_lengths)

                # Debug logging for first iteration to inspect output structure.
                if i == 0:
                    #logger.debug("Model output type: %s", type(model_output))
                    if isinstance(model_output, tuple):
                        #logger.debug("Model output tuple length: %d", len(model_output))
                        for j, item in enumerate(model_output):
                            #logger.debug("Item %d type: %s", j, type(item))
                    elif isinstance(model_output, dict):
                        #logger.debug("Model output keys: %s", list(model_output.keys()))

                # Safely extract expected outputs.
                if isinstance(model_output, tuple) and len(model_output) >= 3:
                    seq_out, curr_att, ctc_inputs = model_output
                elif isinstance(model_output, dict):
                    seq_out = model_output.get('logits', model_output.get('seq_out'))
                    curr_att = model_output.get('attention', model_output.get('attentions', {}))
                    ctc_inputs = model_output.get('ctc_logits', model_output.get('ctc_inputs'))
                else:
                    seq_out = model_output
                    curr_att = {}
                    ctc_inputs = None

                running_att = curr_att

                # Compute CrossEntropy loss.
                ce_loss = self.ce_criterion(seq_out.reshape(-1, seq_out.size(-1)),
                                            targets_golden.reshape(-1))

                # Compute CTC loss if enabled.
                if self.ctc_weight > 0 and ctc_inputs is not None and not isinstance(ctc_inputs, dict):
                    try:
                        ctc_loss = self.ctc_criterion(
                            ctc_inputs.permute(1, 0, 2),
                            targets_golden,
                            feat_lengths,
                            transcript_lengths
                        )
                        loss = ce_loss + self.ctc_weight * ctc_loss
                    except Exception as e:
                        #logger.error("Error in CTC loss calculation: %s", e)
                        ctc_loss = torch.tensor(0.0)
                        loss = ce_loss
                else:
                    ctc_loss = torch.tensor(0.0)
                    loss = ce_loss

            batch_tokens = transcript_lengths.sum().item()
            total_tokens += batch_tokens
            running_ce_loss += ce_loss.item() * batch_tokens
            if self.ctc_weight > 0:
                running_ctc_loss += ctc_loss.item() * batch_tokens
            running_joint_loss += loss.item() * batch_tokens

            # Normalize loss by gradient accumulation steps.
            loss = loss / self.config['training']['gradient_accumulation_steps']

            # Backward pass using scaler.
            self.scaler.scale(loss).backward()

            # Update model weights after accumulating enough gradients.
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            avg_ce_loss = running_ce_loss / total_tokens
            avg_ctc_loss = running_ctc_loss / total_tokens
            avg_joint_loss = running_joint_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_ce_loss))
            batch_bar.set_postfix(
                ce_loss=f"{avg_ce_loss:.4f}",
                ctc_loss=f"{avg_ctc_loss:.4f}",
                joint_loss=f"{avg_joint_loss:.4f}",
                perplexity=f"{perplexity:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/"
                         f"{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Cleanup
            del feats, targets_shifted, targets_golden, feat_lengths, transcript_lengths
            del model_output, seq_out, curr_att, loss
            if ctc_inputs is not None and not isinstance(ctc_inputs, dict):
                del ctc_inputs
            torch.cuda.empty_cache()

        # Step for remaining gradients if any.
        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        avg_ce_loss = running_ce_loss / total_tokens
        avg_ctc_loss = running_ctc_loss / total_tokens
        avg_joint_loss = running_joint_loss / total_tokens
        avg_perplexity_token = torch.exp(torch.tensor(avg_ce_loss))
        avg_perplexity_char = torch.exp(torch.tensor(avg_ce_loss / dataloader.dataset.get_avg_chars_per_token()))
        batch_bar.close()

        return {
            'ce_loss': avg_ce_loss,
            'ctc_loss': avg_ctc_loss,
            'joint_loss': avg_joint_loss,
            'perplexity_token': avg_perplexity_token.item(),
            'perplexity_char': avg_perplexity_char.item()
        }, running_att

    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.

        Args:
            dataloader: DataLoader for validation data.
        Returns:
            Tuple of validation metrics and recognition results.
        """
        results = self.recognize(dataloader)
        references = [r['target'] for r in results]
        hypotheses = [r['generated'] for r in results]
        metrics = self._calculate_asr_metrics(references, hypotheses)
        return metrics, results

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop for ASR training.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            epochs: Number of epochs to train.
        """
        if self.scheduler is None:
            raise ValueError("Scheduler is not initialized, initialize it first!")
        if self.optimizer is None:
            raise ValueError("Optimizer is not initialized, initialize it first!")

        self.text_max_len = max(val_dataloader.dataset.text_max_len, train_dataloader.dataset.text_max_len)

        best_val_loss = float('inf')
        best_val_wer  = float('inf')
        best_val_cer  = float('inf')
        best_val_dist = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            train_metrics, train_attn = self._train_epoch(train_dataloader)
            val_metrics, val_results = self._validate_epoch(val_dataloader)

            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['cer'])

            metrics = {'train': train_metrics, 'val': val_metrics}
            self._log_metrics(metrics, epoch)

            # Save attention plots if available.
            train_attn_keys = list(train_attn.keys())
            if train_attn_keys:
                decoder_self_keys  = [k for k in train_attn_keys if 'dec_self' in k]
                decoder_cross_keys = [k for k in train_attn_keys if 'dec_cross' in k]
                if decoder_self_keys:
                    first_self_key = decoder_self_keys[0]
                    if first_self_key in train_attn:
                        self._save_attention_plot(train_attn[first_self_key][0], epoch, "decoder_self")
                if decoder_cross_keys:
                    last_cross_key = decoder_cross_keys[-1]
                    if last_cross_key in train_attn:
                        self._save_attention_plot(train_attn[last_cross_key][0], epoch, "decoder_cross")

            self.save_generated_text(val_results, f'val_epoch{epoch}')
            self.save_checkpoint('checkpoint-last-epoch-model.pth')

            if val_metrics['cer'] < best_val_cer:
                best_val_cer = val_metrics['cer']
                self.best_metric = val_metrics['cer']
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1

    def evaluate(self, dataloader, max_length: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on a test set using multiple recognition configurations.

        Args:
            dataloader: DataLoader for test data.
            max_length: Optional maximum length for the generated sequence.
        Returns:
            Dictionary mapping recognition configuration names to a DataFrame with evaluation results.
        """
        recognition_configs = self._get_evaluation_recognition_configs()

        eval_results = {}
        for config_name, config in recognition_configs.items():
            try:
                #logger.info("Evaluating with %s config", config_name)
                results = self.recognize(dataloader, config, config_name, max_length)
                generated = [r['generated'] for r in results]
                results_df = pd.DataFrame({
                    'id': range(len(generated)),
                    'transcription': generated
                })
                eval_results[config_name] = results_df
                self.save_generated_text(results, f'test{config_name}_results')
            except Exception as e:
                #logger.error("Error evaluating with %s config: %s", config_name, e)
                continue

        return eval_results

    def recognize(self, dataloader, recognition_config: Optional[Dict[str, Any]] = None,
                  config_name: Optional[str] = None, max_length: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate transcriptions from audio features.

        Args:
            dataloader: DataLoader with evaluation data.
            recognition_config: Dictionary with recognition parameters.
            config_name: An optional name for the recognition configuration.
            max_length: Optional maximum length of the generated sequence.
        Returns:
            List of dictionaries with keys 'generated', 'score', and 'target' (if available).
        """
        if max_length is None and not hasattr(self, 'text_max_len'):
            raise ValueError("text_max_len is not set. Please run training loop first or provide a max_length")

        if recognition_config is None:
            recognition_config = {
                'num_batches': 5,
                'beam_width': 1,
                'temperature': 1.0,
                'repeat_penalty': 1.0,
                'lm_weight': 0.0,
                'lm_model': None
            }
            config_name = 'greedy'

        if recognition_config.get('lm_model') is not None:
            recognition_config['lm_model'].eval()
            recognition_config['lm_model'].to(self.device)

        generator = SequenceGenerator(
            score_fn=None,
            tokenizer=self.tokenizer,
            max_length=max_length if max_length is not None else self.text_max_len,
            device=self.device
        )

        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False,
                         position=0, desc=f"[Recognizing ASR] : {config_name}")
        results = []

        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                feats, _, targets_golden, feat_lengths, _ = batch
                feats = feats.to(self.device)
                feat_lengths = feat_lengths.to(self.device)
                if targets_golden is not None:
                    targets_golden = targets_golden.to(self.device)

                encoder_output, pad_mask_src, _, _ = self.model.encode(feats, feat_lengths)

                def get_score(x):
                    asr_logits = self.model.score(x, encoder_output, pad_mask_src)
                    if recognition_config.get('lm_model') is not None:
                        lm_logits = recognition_config['lm_model'].score(x)
                        return asr_logits + recognition_config['lm_weight'] * lm_logits
                    return asr_logits

                generator.score_fn = get_score

                batch_size = feats.size(0)
                prompts = torch.full((batch_size, 1), self.tokenizer.sos_id,
                                      dtype=torch.long, device=self.device)

                # Use beam search if beam_width > 1.
                if recognition_config['beam_width'] > 1:
                    #logger.debug("Beam search activated with beam width: %d", recognition_config['beam_width'])
                    seqs, scores = generator.generate_beam(
                        prompts,
                        beam_width=recognition_config['beam_width'],
                        temperature=recognition_config['temperature'],
                        repeat_penalty=recognition_config['repeat_penalty']
                    )
                    #logger.debug("Beam search output shapes: %s, %s", seqs.shape, scores.shape)
                    seqs = seqs[:, 0, :]
                    scores = scores[:, 0]
                else:
                    seqs, scores = generator.generate_greedy(prompts,
                                                              temperature=recognition_config['temperature'])
                    #logger.debug("Greedy search output shape: %s", seqs.shape)

                del feats, feat_lengths, encoder_output, pad_mask_src, prompts
                torch.cuda.empty_cache()

                post_processed_preds = generator.post_process_sequence(seqs, self.tokenizer)

                if targets_golden is not None:
                    post_processed_targets = generator.post_process_sequence(targets_golden, self.tokenizer)
                    for j, (pred, target) in enumerate(zip(post_processed_preds, post_processed_targets)):
                        results.append({
                            'target': self.tokenizer.decode(target.tolist(), skip_special_tokens=True),
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item()
                        })
                else:
                    for j, pred in enumerate(post_processed_preds):
                        results.append({
                            'generated': self.tokenizer.decode(pred.tolist(), skip_special_tokens=True),
                            'score': scores[j].item()
                        })

                batch_bar.update()

                if recognition_config['num_batches'] is not None and i >= recognition_config['num_batches'] - 1:
                    break

            batch_bar.close()
            return results

    def _get_evaluation_recognition_configs(self, lm_model: Optional[DecoderOnlyTransformer] = None,
                                              lm_weight: float = 0.0) -> Dict[str, Dict[str, Any]]:
        """
        Retrieve several recognition configurations for evaluation.
        """
        common_config = {
            'num_batches': None,
            'temperature': 1.0,
            'repeat_penalty': 1.0,
            'lm_weight': lm_weight,
            'lm_model': lm_model
        }
        greedy_config = common_config.copy()
        greedy_config.update({'beam_width': 1})

        beam_10_config = common_config.copy()
        beam_10_config.update({'beam_width': 10})

        beam_20_config = common_config.copy()
        beam_20_config.update({'beam_width': 20})

        return {
            'greedy': greedy_config,
            'beam_10': beam_10_config,
            'beam_20': beam_20_config
        }

    def _calculate_asr_metrics(self, references: Union[str, List[str]],
                               hypotheses: Union[str, List[str]]) -> Dict[str, float]:
        """
        Calculate edit distance, WER, and CER for ASR results.
        """
        wer_metric = tmt.WordErrorRate()
        word_edit_metric = tmt.EditDistance(reduction='mean')
        cer_metric = tmt.CharErrorRate()

        word_dist = word_edit_metric(hypotheses, references)
        wer = wer_metric(hypotheses, references)
        cer = cer_metric(hypotheses, references)

        return {
            'word_dist': word_dist.item(),
            'wer': wer.item() * 100,
            'cer': cer.item() * 100
        }


class ProgressiveTrainer(ASRTrainer):
    """
    Progressive Trainer class that implements curriculum learning for ASR training.

    This trainer extends ASRTrainer to support stage-based training, gradual unfreezing,
    dynamic data subsetting, and a smooth transition to full model training.
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.current_stage = 0
        self.all_encoder_layers = list(self.model.enc_layers)
        self.all_decoder_layers = list(self.model.dec_layers)

    def configure_stage(self, stage_config):
        """Configure model for the current training stage."""
        header = "=" * 80
        #logger.info("\n%s\n%20s\n%s", header, f"Starting Stage: {stage_config['name']}".center(80), header)

        #logger.info("Configuration Details:")
        #logger.info("├── Data Subset: %.1f%% of training data", stage_config['data_subset'] * 100)
        #logger.info("├── Training Epochs: %d", stage_config['epochs'])
        #logger.info("├── Dropout: %.4f", stage_config['dropout'])
        #logger.info("├── Label Smoothing: %.4f", stage_config['label_smoothing'])

        self.model.dropout.p = stage_config['dropout']
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=stage_config['label_smoothing']
        )

        encoder_freeze = stage_config.get('encoder_freeze', [])
        decoder_freeze = stage_config.get('decoder_freeze', [])

        encoder_active_layers = stage_config['encoder_active_layers']
        if encoder_freeze and len(encoder_freeze) != len(encoder_active_layers):
            raise ValueError(f"Encoder freeze list length ({len(encoder_freeze)}) must match number of active encoder layers ({len(encoder_active_layers)})")

        self.model.enc_layers = nn.ModuleList([
            self.all_encoder_layers[i] for i in encoder_active_layers
        ])
        self.model.num_encoder_layers = len(encoder_active_layers)

        decoder_active_layers = stage_config['decoder_active_layers']
        if decoder_freeze and len(decoder_freeze) != len(decoder_active_layers):
            raise ValueError(f"Decoder freeze list length ({len(decoder_freeze)}) must match number of active decoder layers ({len(decoder_active_layers)})")

        self.model.dec_layers = nn.ModuleList([
            self.all_decoder_layers[i] for i in decoder_active_layers
        ])
        self.model.num_decoder_layers = len(decoder_active_layers)

        frozen_count = 0
        trainable_count = 0

        #logger.info("├── Encoder Layers:")
        for idx, layer in enumerate(self.model.enc_layers):
            should_freeze = encoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            #logger.info("│   ├── Layer %d: %s", encoder_active_layers[idx], "Frozen" if should_freeze else "Trainable")

        #logger.info("├── Decoder Layers:")
        for idx, layer in enumerate(self.model.dec_layers):
            should_freeze = decoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            #logger.info("│   ├── Layer %d: %s", decoder_active_layers[idx], "Frozen" if should_freeze else "Trainable")

        #logger.info("├── Frozen Parameters: %d", frozen_count)
        #logger.info("└── Trainable Parameters: %d", trainable_count)

    def progressive_train(self, train_dataloader, val_dataloader, stages: List[Dict[str, Any]]):
        """
        Perform progressive training through defined stages.

        Args:
            train_dataloader: DataLoader for training data.
            val_dataloader: DataLoader for validation data.
            stages: List of stage configurations.
        """
        for stage_idx, stage_config in enumerate(stages):
            self.current_stage = stage_idx
            self.configure_stage(stage_config)
            subset_train_dataloader = self.get_subset_dataloader(train_dataloader, stage_config['data_subset'])
            super().train(subset_train_dataloader, val_dataloader, epochs=stage_config['epochs'])

    def transition_to_full_training(self):
        """Transition to full training by restoring all model layers and unfreezing parameters."""
        #logger.info("Transitioning to full training.")
        self.model.enc_layers = nn.ModuleList(self.all_encoder_layers)
        self.model.dec_layers = nn.ModuleList(self.all_decoder_layers)
        self.model.num_encoder_layers = len(self.all_encoder_layers)
        self.model.num_decoder_layers = len(self.all_decoder_layers)

        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss']['label_smoothing']
        )

        unfrozen_count = 0
        for param in self.model.parameters():
            param.requires_grad = True
            unfrozen_count += param.numel()
        #logger.info("Total Unfrozen Parameters: %d", unfrozen_count)
        self.best_metric = float('inf')

    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Run full training phase (transition to full training then call parent train).

        Args:
            train_dataloader: DataLoader for training.
            val_dataloader: DataLoader for validation.
            epochs: Number of epochs to run.
        """
        self.transition_to_full_training()
        super().train(train_dataloader, val_dataloader, epochs=epochs)

    def get_subset_dataloader(self, dataloader, subset_fraction):
        """
        Create a DataLoader containing only a fraction of the original dataset.

        Args:
            dataloader: Original DataLoader.
            subset_fraction: Fraction of data to keep.
        Returns:
            A new DataLoader with the subset.
        """
        dataset = dataloader.dataset
        total_samples = len(dataset)
        subset_size = int(total_samples * subset_fraction)
        indices = torch.randperm(total_samples)[:subset_size]
        subset_dataset = Subset(dataset, indices)
        subset_dataset.text_max_len = dataset.text_max_len
        subset_dataset.feat_max_len = dataset.feat_max_len
        subset_dataset.get_avg_chars_per_token = dataset.get_avg_chars_per_token

        subset_loader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['NUM_WORKERS'],
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
        return subset_loader