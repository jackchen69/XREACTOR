import math
import random
from einops import rearrange
import torch
from framework.modules.post_processor import Processor
from framework.utils.compute_metrics import compute_metrics
from framework.utils.util import from_pretrained_checkpoint
from utils.util import AverageMeter, get_lr
from omegaconf import DictConfig
from tqdm import tqdm
from hydra.utils import instantiate
from torch.utils.tensorboard import SummaryWriter
import logging

import os
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from mreactor.mfg import MFG, MFGConfig
except Exception:
    from MFG import MFG, MFGConfig


class Trainer:
    def __init__(self,
                 resumed_training: bool = False,
                 generic: DictConfig = None,
                 renderer: DictConfig = None,
                 model: DictConfig = None,
                 criterion: DictConfig = None,
                 **kwargs):
       
        super().__init__()
        self.resumed_training = resumed_training
        self.renderer = renderer
        self.model_cfg = model
        self.criterion_cfg = criterion

        if torch.cuda.device_count() > 0:
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.device = device
        self.kwargs = kwargs
        self.trainer_cfg = generic
        self.optim_cfg = kwargs.pop("optim")
        self.task = kwargs.get("task")

        self.use_mfg: bool = kwargs.get("use_mfg", True)                 
        self.mfg_out_dim: int = kwargs.get("mfg_out_dim", 25)            
        self.mfg_num_preds: int = kwargs.get("mfg_num_preds", 3)         
        self.mfg_joint_train: bool = kwargs.get("mfg_joint_train", False)
        self.mfg_loss_weight: float = kwargs.get("mfg_loss_weight", 1.0) 
        self.mfg_log_attn_every: int = kwargs.get("mfg_log_attn_every", 100)  
        self.mfg_save_dir: str = kwargs.get("mfg_save_dir", "mfg_logs")      
        self.mfg_hist_tb: bool = kwargs.get("mfg_hist_tb", True)        
        self.mfg: MFG | None = None

        try:
            Path(self.mfg_save_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def set_data_module(self, data_module):
        self.data_module = data_module

    def data_resample(self,
                      speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips,
                      listener_video_clips, listener_emotion_clips, listener_3dmm_clips,
                      speaker_seq_lengths, listener_seq_lengths, speaker_audio2text_clips, speaker_lip_clips):

        s_ratio = self.trainer_cfg.s_ratio
        window_size = self.trainer_cfg.window_size
        clip_length = self.trainer_cfg.clip_length
        s_window_size = s_ratio * window_size
        l_window_size = window_size

        if self.task == 'offline':
            stack = lambda clips: torch.stack(clips, dim=0)
            speaker_audio, speaker_emotion, speaker_3dmm, speaker_audio2text, speaker_lip = (
                stack(clips) for clips in (speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips, speaker_audio2text_clips, speaker_lip_clips))
            listener_video, listener_emotion, listener_3dmm = (
                stack(clips) for clips in (listener_video_clips, listener_emotion_clips, listener_3dmm_clips))
            past_listener_emotion = past_listener_3dmm = None
            seq_lengths = torch.tensor(speaker_seq_lengths).clamp(max=clip_length)

        elif self.task == "online":
            def get_padded(clip: torch.Tensor, length: int, target_len: int) -> torch.Tensor:
                clip = clip[:length]
                if length < target_len:
                    pad_shape = (target_len - length, *clip.shape[1:])
                    clip = torch.cat([clip, clip.new_zeros(pad_shape)], dim=0)
                return clip

            speaker_audio, speaker_emotion, speaker_3dmm, speaker_audio2text, speaker_lip = [], [], [], []. []
            listener_video, listener_emotion, listener_3dmm = [], [], []
            past_listener_emotion, past_listener_3dmm = [], []

            for (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip, speaker_seq_length,
                 listener_video_clip, listener_emotion_clip, listener_3dmm_clip, listener_seq_length, speaker_audio2text_clip, speaker_lip_clip) in \
                    zip(speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips, speaker_seq_lengths,
                        listener_video_clips, listener_emotion_clips, listener_3dmm_clips, listener_seq_lengths, speaker_audio2text_clips, speaker_lip_clips):
                seq_length = speaker_seq_length
                assert speaker_seq_length == listener_seq_length, "Sequence length not equal"

                speaker_audio_clip = get_padded(speaker_audio_clip, seq_length, s_window_size)
                speaker_emotion_clip = get_padded(speaker_emotion_clip, seq_length, s_window_size)
                speaker_3dmm_clip = get_padded(speaker_3dmm_clip, seq_length, s_window_size)
                listener_video_clip = get_padded(listener_video_clip, seq_length, s_window_size)
                listener_emotion_clip = get_padded(listener_emotion_clip, seq_length, s_window_size)
                listener_3dmm_clip = get_padded(listener_3dmm_clip, seq_length, s_window_size)
                speaker_audio2text_clip = get_padded(speaker_audio2text_clip, seq_length, s_window_size)
                speaker_lip_clip = get_padded(speaker_lip_clip, seq_length, s_window_size)


                if seq_length < clip_length:
                    cp = random.randint(0, seq_length - s_window_size) if seq_length > s_window_size else 0
                else:
                    cp = random.randint(0, clip_length - s_window_size)

                du = cp + s_window_size
                speaker_audio_clip = speaker_audio_clip[cp:du]
                speaker_emotion_clip = speaker_emotion_clip[cp:du]
                speaker_3dmm_clip = speaker_3dmm_clip[cp:du]
                listener_video_clip = listener_video_clip[du - l_window_size:du]
                past_listener_emotion_clip = listener_emotion_clip[(du - 2 * l_window_size): (du - l_window_size)]
                listener_emotion_clip = listener_emotion_clip[(du - l_window_size): du]
                past_listener_3dmm_clip = listener_3dmm_clip[(du - 2 * l_window_size): (du - l_window_size)]
                listener_3dmm_clip = listener_3dmm_clip[(du - l_window_size): du]
                speaker_audio2text_clip = speaker_audio2text_clip[cp:du]
                speaker_lip_clip = speaker_lip_clip[cp:du]

                speaker_audio.append(speaker_audio_clip)
                speaker_emotion.append(speaker_emotion_clip)
                speaker_3dmm.append(speaker_3dmm_clip)
                listener_video.append(listener_video_clip)
                listener_emotion.append(listener_emotion_clip)
                listener_3dmm.append(listener_3dmm_clip)
                past_listener_emotion.append(past_listener_emotion_clip)
                past_listener_3dmm.append(past_listener_3dmm_clip)
                speaker_audio2text.append(speaker_audio2text_clip)
                speaker_lip.append(speaker_lip_clip)


            speaker_audio = torch.stack(speaker_audio, dim=0)  
            speaker_emotion = torch.stack(speaker_emotion, dim=0)  
            speaker_3dmm = torch.stack(speaker_3dmm, dim=0)  
            listener_video = torch.stack(listener_video, dim=0)  
            listener_emotion = torch.stack(listener_emotion, dim=0) 
            listener_3dmm = torch.stack(listener_3dmm, dim=0)  
            past_listener_emotion = torch.stack(past_listener_emotion, dim=0)  
            past_listener_3dmm = torch.stack(past_listener_3dmm, dim=0)  
            speaker_audio2text = torch.stack(speaker_audio2text, dim=0)  
            speaker_lip = torch.stack(speaker_lip, dim=0)  
            seq_lengths = None
        else:
            raise ValueError("Unknown task type")

        return (speaker_audio, speaker_emotion, speaker_3dmm, listener_video, listener_emotion,
                listener_3dmm, past_listener_emotion, past_listener_3dmm, seq_lengths, speaker_audio2text, speaker_lip)

    def fit(self):
        """
        # relative directory
        root_dir = save/${trainer.task_name}/${data.data_name}/${folder_name}
        # absolute directory
        saving_dir = Path(hydra.utils.to_absolute_path(root_dir))
        # get saving path
        saving_path = str(saving_dir / ...)
        """

        self.start_epoch = self.trainer_cfg.start_epoch
        self.epochs = self.trainer_cfg.epochs
        self.tb_dir = self.trainer_cfg.tb_dir
        self.clip_grad = self.trainer_cfg.clip_grad
        self.val_period = self.trainer_cfg.val_period
        stage = "fit"

        logger.info("Loading data module")
        self.train_loader, self.val_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Data module loaded")

        logger.info("Loading criterion")
        self.criterion = instantiate(self.criterion_cfg)
        logger.info("Criterion loaded")

        logger.info("Loading writer")
        self.writer = SummaryWriter(self.tb_dir)
        logger.info(f"Writer loaded: {self.tb_dir}")
        self.main_diffusion(stage)

    def main_diffusion(self, stage):
        model = instantiate(self.model_cfg.diff_model,
                            stage=stage,
                            resumed_training=self.resumed_training,
                            latent_embedder=self.model_cfg.latent_embedder \
                                if hasattr(self.model_cfg, "latent_embedder") else None,
                            audio_encoder=self.model_cfg.audio_encoder \
                                if hasattr(self.model_cfg, "audio_encoder") else None,
                            **self.kwargs,
                            _recursive_=False)
        model.to(self.device)

        if self.use_mfg and self.mfg is None:
            mfg_cfg = MFGConfig(
                d_model=256,           
                d_fine=64,
                d_coarse=128,
                d_out=self.mfg_out_dim,
                n_heads=4,
                N_variants=self.mfg_num_preds
            )
            self.mfg = MFG(mfg_cfg).to(self.device)

        optimizer = instantiate(self.optim_cfg, lr=self.trainer_cfg.lr, params=model.parameters())

        if self.use_mfg and self.mfg is not None:
            optimizer.add_param_group({"params": self.mfg.parameters(), "lr": self.trainer_cfg.lr})

        if self.resumed_training:
            checkpoint_path = model.get_ckpt_path(model.diffusion_decoder.model, runid="resume_runid", last=True)
            best_diff_decoder_loss, self.start_epoch = (
                from_pretrained_checkpoint(checkpoint_path, optimizer, self.device)
            )
            logger.info(f"Resume training from epoch {self.start_epoch}")
        else:
            best_diff_decoder_loss = float('inf')
        print(f"Best validation loss: {best_diff_decoder_loss}")

        scheduler = instantiate(self.kwargs.pop("scheduler"), optimizer, len(self.train_loader))

        for epoch in range(self.start_epoch, self.epochs):
            diff_decoder_loss, au_rec_loss, va_rec_loss, em_rec_loss = (
                self.train_diffusion(model, self.train_loader, optimizer, scheduler,
                                     self.criterion, epoch, self.writer, self.device))
            logging.info(f"Epoch: {epoch + 1}  train_diff_loss: {diff_decoder_loss:.5f}  au_rec_loss: {au_rec_loss:.5f}"
                         f"  va_rec_loss: {va_rec_loss:.5f}  em_rec_loss: {em_rec_loss:.5f}")

            if (epoch + 1) % self.val_period == 0:
                diff_decoder_loss, au_rec_loss, va_rec_loss, em_rec_loss = (
                    self.val_diffusion(model, self.val_loader, self.criterion, self.device))
                logging.info(f"Epoch: {epoch + 1}  val_diff_loss: {diff_decoder_loss:.5f}  au_rec_loss: {au_rec_loss:.5f}"
                             f"  va_rec_loss: {va_rec_loss:.5f}  em_rec_loss: {em_rec_loss:.5f}")

                if diff_decoder_loss < best_diff_decoder_loss:
                    best_diff_decoder_loss = diff_decoder_loss
                    logging.info(
                        f"New best diff_decoder_loss ({best_diff_decoder_loss:.5f}) at epoch {epoch + 1}, "
                        f"saving checkpoint"
                    )
                    model.save_ckpt(optimizer, best=True, epoch=(epoch+1), best_loss=best_diff_decoder_loss)

                model.save_ckpt(optimizer, epoch=(epoch + 1), best_loss=best_diff_decoder_loss)
                model.save_ckpt(optimizer, last=True, epoch=(epoch+1), best_loss=best_diff_decoder_loss)

    def train_diffusion(self, model, data_loader, optimizer, scheduler,
                        criterion, epoch, writer, device):
        whole_losses = AverageMeter()
        au_rec_losses = AverageMeter()
        va_rec_losses = AverageMeter()
        em_rec_losses = AverageMeter()

        model.train()
        step_global = epoch * len(data_loader)

        for batch_idx, (
                speaker_audio_clip,
                speaker_video_clip,
                speaker_emotion_clip,
                speaker_3dmm_clip,
                listener_video_clip,
                listener_emotion_clip,
                listener_3dmm_clip,
                speaker_clip_length,
                listener_clip_length,
                speaker_audio2text_clip,
                speaker_lip_clip,
        ) in enumerate(tqdm(data_loader)):

            (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip,
             listener_video_clip, listener_emotion_clip, listener_3dmm_clip,
             past_listener_emotion, past_listener_3dmm, motion_lengths, speaker_audio2text_clip, speaker_lip_clip,) = self.data_resample(
                    speaker_audio_clips=speaker_audio_clip, speaker_emotion_clips=speaker_emotion_clip,
                    speaker_3dmm_clips=speaker_3dmm_clip, listener_video_clips=listener_video_clip,
                    listener_emotion_clips=listener_emotion_clip, listener_3dmm_clips=listener_3dmm_clip,
                    speaker_seq_lengths=speaker_clip_length, listener_seq_lengths=listener_clip_length,speaker_audio2text_clips=speaker_audio2text_clip,
                    speaker_lip_clips=speaker_lip_clip)

            (speaker_audio_clip,  
             speaker_emotion_clip, 
             speaker_3dmm_clip,  
             listener_video_clip,
             listener_emotion_clip,  
             speaker_audio2text_clip, 
             speaker_lip_clip, 
             ) = (speaker_audio_clip.to(device),
                 speaker_emotion_clip.to(device),
                 speaker_3dmm_clip.to(device),
                 listener_video_clip.to(device),
                 listener_emotion_clip.to(device),
                 speaker_audio2text_clip.to(device),
                 speaker_lip_clip.to(device))
            batch_size = speaker_audio_clip.shape[0]

            outputs = model(
                speaker_audio_input=speaker_audio_clip,
                speaker_emotion_input=speaker_emotion_clip,
                speaker_3dmm_input=speaker_3dmm_clip,
                listener_emotion_input=listener_emotion_clip,
                past_listener_emotion=past_listener_emotion,
                motion_length=motion_lengths,
                speaker_audio2text_input=speaker_audio2text_clip,
                speaker_lip_input=speaker_lip_clip,
            )

            output = criterion(outputs)
            loss = output["loss"]

            iteration = batch_idx + len(data_loader) * epoch
            if writer is not None:
                writer.add_scalar("Train/loss", loss.data.item(), iteration)

            if self.use_mfg and self.mfg is not None:
                P_gt_mfg = listener_emotion_clip if self.mfg_out_dim == 25 else listener_3dmm_clip
                motion_len_mfg = motion_lengths if torch.is_tensor(motion_lengths) else None

                mfg_out = self.mfg(
                    Z_f=speaker_emotion_clip,          
                    Z_c=speaker_3dmm_clip,             
                    L_text=speaker_audio2text_clip,    
                    lengths=motion_len_mfg,
                    P_gt=P_gt_mfg,
                    return_losses=True,
                    return_attn=True                   
                )

                if writer is not None:
                    for k in ["loss", "L_v", "L_g"]:
                        if k in mfg_out and torch.is_tensor(mfg_out[k]):
                            writer.add_scalar(f"Train/MFG/{k}", float(mfg_out[k].detach().cpu()), iteration)

                if self.mfg_hist_tb and iteration % max(1, self.mfg_log_attn_every) == 0:
                    for tag_key, hist_name in [
                        ("V_rel", "MFG/values_relative"),
                        ("V_subj", "MFG/values_subject"),
                    ]:
                        if tag_key in mfg_out and torch.is_tensor(mfg_out[tag_key]):
                            try:
                                writer.add_histogram(hist_name, mfg_out[tag_key].detach().cpu(), iteration)
                            except Exception:
                                pass
                    for tag_key, hist_name in [
                        ("A_rel", "MFG/weights_relative"),
                        ("A_subj", "MFG/weights_subject"),
                    ]:
                        if tag_key in mfg_out and torch.is_tensor(mfg_out[tag_key]):
                            try:
                                writer.add_histogram(hist_name, mfg_out[tag_key].detach().cpu(), iteration)
                            except Exception:
                                pass

                    try:
                        torch.save(
                            {
                                "iter": int(iteration),
                                "epoch": int(epoch),
                                "V_rel": mfg_out.get("V_rel", None).detach().cpu() if isinstance(mfg_out.get("V_rel", None), torch.Tensor) else None,
                                "V_subj": mfg_out.get("V_subj", None).detach().cpu() if isinstance(mfg_out.get("V_subj", None), torch.Tensor) else None,
                                "A_rel": mfg_out.get("A_rel", None).detach().cpu() if isinstance(mfg_out.get("A_rel", None), torch.Tensor) else None,
                                "A_subj": mfg_out.get("A_subj", None).detach().cpu() if isinstance(mfg_out.get("A_subj", None), torch.Tensor) else None,
                            },
                            os.path.join(self.mfg_save_dir, f"train_attn_values_step_{int(iteration)}.pt")
                        )
                    except Exception as e:
                        logger.debug(f"[MFG] save attn values failed at iter {iteration}: {e}")

                if self.mfg_joint_train and "loss" in mfg_out:
                    loss = loss + self.mfg_loss_weight * mfg_out["loss"]

            whole_losses.update(loss.data.item(), batch_size)
            au_rec_losses.update(output["loss_au"].data.item(), batch_size)
            va_rec_losses.update(output["loss_va"].data.item(), batch_size)
            em_rec_losses.update(output["loss_em"].data.item(), batch_size)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if scheduler is not None and (epoch + 1) >= 5:
            scheduler.step()
        lr = get_lr(optimizer=optimizer)
        if writer is not None:
            writer.add_scalar("Train/lr", lr, epoch)

        return whole_losses.avg, au_rec_losses.avg, va_rec_losses.avg, em_rec_losses.avg

    def val_diffusion(self, model, val_loader, criterion, device):
        whole_losses = AverageMeter()
        au_rec_losses = AverageMeter()
        va_rec_losses = AverageMeter()
        em_rec_losses = AverageMeter()

        model.eval()
        for batch_idx, (
                speaker_audio_clip,
                speaker_video_clip,
                speaker_emotion_clip,
                speaker_3dmm_clip,
                listener_video_clip,
                listener_emotion_clip,
                listener_3dmm_clip,
                speaker_clip_length,
                listener_clip_length,
                speaker_audio2text_clip,
                speaker_lip_clip,

        ) in enumerate(tqdm(val_loader)):
            (speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip,
             listener_video_clip, listener_emotion_clip, listener_3dmm_clip,
             past_listener_emotion, past_listener_3dmm, motion_lengths, speaker_audio2text_clip, speaker_lip_clip) = self.data_resample(
                    speaker_audio_clips=speaker_audio_clip, speaker_emotion_clips=speaker_emotion_clip,
                    speaker_3dmm_clips=speaker_3dmm_clip, listener_video_clips=listener_video_clip,
                    listener_emotion_clips=listener_emotion_clip, listener_3dmm_clips=listener_3dmm_clip,
                    speaker_seq_lengths=speaker_clip_length, listener_seq_lengths=listener_clip_length,speaker_audio2text_clips=speaker_audio2text_clip,
                    speaker_lip_clips=speaker_lip_clip)

            (speaker_audio_clip, 
             speaker_emotion_clip,  
             speaker_3dmm_clip, 
             listener_video_clip,
             listener_emotion_clip,  
             speaker_audio2text_clip, 
             speaker_lip_clip,  
             ) = (speaker_audio_clip.to(device),
                 speaker_emotion_clip.to(device),
                 speaker_3dmm_clip.to(device),
                 listener_video_clip.to(device),
                 listener_emotion_clip.to(device),
                 speaker_audio2text_clip.to(device),
                 speaker_lip_clip.to(device))
            batch_size = speaker_audio_clip.shape[0]

            with torch.no_grad():
                outputs = model(
                    speaker_audio_input=speaker_audio_clip,
                    speaker_emotion_input=speaker_emotion_clip,
                    speaker_3dmm_input=speaker_3dmm_clip,
                    listener_emotion_input=listener_emotion_clip,
                    past_listener_emotion=past_listener_emotion,
                    motion_length=motion_lengths,
                    speaker_audio2text_input=speaker_audio2text_clip,
                    speaker_lip_input=speaker_lip_clip,

                )

                output = criterion(outputs)
                loss = output["loss"]

                if self.use_mfg and self.mfg is not None:
                    P_gt_mfg = listener_emotion_clip if self.mfg_out_dim == 25 else listener_3dmm_clip
                    motion_len_mfg = motion_lengths if torch.is_tensor(motion_lengths) else None
                    mfg_out = self.mfg(
                        Z_f=speaker_emotion_clip,
                        Z_c=speaker_3dmm_clip,
                        L_text=speaker_audio2text_clip,
                        lengths=motion_len_mfg,
                        P_gt=P_gt_mfg,
                        return_losses=False,
                        return_attn=True
                    )
                    try:
                        torch.save(
                            {
                                "iter": int(batch_idx),
                                "V_rel": mfg_out.get("V_rel", None).detach().cpu() if isinstance(mfg_out.get("V_rel", None), torch.Tensor) else None,
                                "V_subj": mfg_out.get("V_subj", None).detach().cpu() if isinstance(mfg_out.get("V_subj", None), torch.Tensor) else None,
                                "A_rel": mfg_out.get("A_rel", None).detach().cpu() if isinstance(mfg_out.get("A_rel", None), torch.Tensor) else None,
                                "A_subj": mfg_out.get("A_subj", None).detach().cpu() if isinstance(mfg_out.get("A_subj", None), torch.Tensor) else None,
                            },
                            os.path.join(self.mfg_save_dir, f"val_attn_values_step_{int(batch_idx)}.pt")
                        )
                    except Exception as e:
                        logger.debug(f"[MFG] save val attn values failed at step {batch_idx}: {e}")

            whole_losses.update(loss.data.item(), batch_size)
            au_rec_losses.update(output["loss_au"].data.item(), batch_size)
            va_rec_losses.update(output["loss_va"].data.item(), batch_size)
            em_rec_losses.update(output["loss_em"].data.item(), batch_size)

        return whole_losses.avg, au_rec_losses.avg, va_rec_losses.avg, em_rec_losses.avg
    
    def test(self):
        stage = "test"
        data_clamp = self.kwargs.pop("data_clamp")
        logger.info("Loading test data module")
        test_loader = self.data_module.get_dataloader(stage=stage)
        logger.info("Test data module loaded")
        clip_len = self.trainer_cfg.clip_length
        w = self.trainer_cfg.window_size
        s_ratio = self.trainer_cfg.s_ratio
        s_w = s_ratio * w

        model = instantiate(self.model_cfg.diff_model,
                            stage=stage,
                            latent_embedder=self.model_cfg.latent_embedder \
                                if hasattr(self.model_cfg, "latent_embedder") else None,
                            audio_encoder=self.model_cfg.audio_encoder \
                                if hasattr(self.model_cfg, "audio_encoder") else None,
                            **self.kwargs,
                            _recursive_=False)
        model.to(self.device)
        model.eval()

        logger.info("Loading post processor")
        post_processor = Processor(config_name=self.kwargs.pop("post_config_name"),
                                   clip_len_test=self.kwargs.pop("post_clip_length"),
                                   device=self.device,)
        logger.info("Post processor loaded")

        GT_listener_emotions_all = []
        pred_listener_emotions_all = []
        input_speaker_emotions_all = []

        test_attn_values = []

        for batch_idx, (
                speaker_audio_clips,
                speaker_video_clips,
                speaker_emotion_clips,
                speaker_3dmm_clips,
                listener_video_clips,
                listener_emotion_clips,
                _,
                speaker_seq_lengths,
                listener_seq_lengths,
                speaker_audio2text_clips,
                speaker_lip_clips,
                
        ) in enumerate(tqdm(test_loader)):

            GT_listener_emotions_all.extend(listener_emotion_clips)
            input_speaker_emotions_all.extend(speaker_emotion_clips)

            clip_batch_size = 8  
            speaker_audios = []
            speaker_emotions = []
            speaker_3dmms = []
            motion_lengths = []
            sample_batch_size = []
            speaker_audio2text_clips_stack = []
            speaker_lip_clips_stack = []

            for speaker_audio_clip, speaker_emotion_clip, speaker_3dmm_clip, speaker_seq_length, speaker_audio2text_clip, speaker_lip_clip  in \
                    zip(speaker_audio_clips, speaker_emotion_clips, speaker_3dmm_clips, speaker_seq_lengths, speaker_audio2text_clips, speaker_lip_clips):
                length = speaker_seq_length

                if self.task == "offline":
                    remain_length = length % clip_len
                    b = math.ceil((length + clip_len - remain_length) / clip_len)
                    lengths = torch.tensor([clip_len] * (b - 1) + [remain_length])
                    sample_batch_size.append(b)

                    speaker_audio_clip = torch.cat((speaker_audio_clip,
                                                    torch.zeros(
                                                        size=(clip_len - remain_length, speaker_audio_clip.shape[-1]))),
                                                   dim=0)
                    speaker_audio_clip = rearrange(speaker_audio_clip, '(b l) d -> b l d', b=b)

                    speaker_lip_clip = torch.cat((speaker_lip_clip,
                                                    torch.zeros(
                                                        size=(clip_len - remain_length, speaker_lip_clip.shape[-1]))),
                                                   dim=0)
                    speaker_lip_clip = rearrange(speaker_lip_clip, '(b l) d -> b l d', b=b)

                    speaker_emotion_clip = torch.cat((speaker_emotion_clip,
                                                      torch.zeros(size=(clip_len - remain_length,
                                                                        speaker_emotion_clip.shape[-1]))), dim=0)
                    speaker_emotion_clip = rearrange(speaker_emotion_clip, '(b l) d -> b l d', b=b)

                    speaker_3dmm_clip = torch.cat((speaker_3dmm_clip,
                                                   torch.zeros(
                                                       size=(clip_len - remain_length, speaker_3dmm_clip.shape[-1]))),
                                                  dim=0)
                    speaker_3dmm_clip = rearrange(speaker_3dmm_clip, '(b l) d -> b l d', b=b)

                    speaker_audio2text_clip = torch.cat((speaker_audio2text_clip,
                                                    torch.zeros(
                                                        size=(clip_len - remain_length, speaker_audio2text_clip.shape[-1]))),
                                                   dim=0)
                    speaker_audio2text_clip = rearrange(speaker_audio2text_clip, '(b l) d -> b l d', b=b)

                    speaker_audios.append(speaker_audio_clip)
                    speaker_emotions.append(speaker_emotion_clip)
                    speaker_3dmms.append(speaker_3dmm_clip)
                    motion_lengths.append(lengths)
                    speaker_audio2text_clips_stack.append(speaker_audio2text_clip)
                    speaker_lip_clips_stack.append(speaker_lip_clip)

                else:  
                    num_windows = math.ceil(length / w)
                    sample_batch_size.append(num_windows)

                    speaker_audio_clip = torch.cat(
                        (torch.zeros(size=((s_w - w), speaker_audio_clip.shape[-1])),
                         speaker_audio_clip,
                         torch.zeros(size=((num_windows * w - length), speaker_audio_clip.shape[-1]))), dim=0)
                    speaker_lip_clip = torch.cat(
                        (torch.zeros(size=((s_w - w), speaker_lip_clip.shape[-1])),
                         speaker_lip_clip,
                         torch.zeros(size=((num_windows * w - length), speaker_lip_clip.shape[-1]))), dim=0)
                    speaker_emotion_clip = torch.cat(
                        (torch.zeros(size=((s_w - w), speaker_emotion_clip.shape[-1])),
                         speaker_emotion_clip,
                         torch.zeros(size=((num_windows * w - length), speaker_emotion_clip.shape[-1]))), dim=0)
                    speaker_3dmm_clip = torch.cat(
                        (torch.zeros(size=((s_w - w), speaker_3dmm_clip.shape[-1])),
                         speaker_3dmm_clip,
                         torch.zeros(size=((num_windows * w - length), speaker_3dmm_clip.shape[-1]))), dim=0)
                    speaker_audio2text_clip = torch.cat(
                        (torch.zeros(size=((s_w - w), speaker_audio2text_clip.shape[-1])),
                         speaker_audio2text_clip,
                         torch.zeros(size=((num_windows * w - length), speaker_audio2text_clip.shape[-1]))), dim=0)

                    motion_length_list = []
                    speaker_audio_clip_list = []
                    speaker_emotion_clip_list = []
                    speaker_3dmm_clip_list = []
                    speaker_audio2text_clip_list = []
                    speaker_lip_clip_list = []

                    for i in range(num_windows):
                        motion_length_list.append(w) if i < num_windows - 1 else motion_length_list.append(
                            length - i * w)
                        speaker_audio_clip_list.append(speaker_audio_clip[i*w: i*w + s_w])
                        speaker_emotion_clip_list.append(speaker_emotion_clip[i*w: i*w + s_w])
                        speaker_3dmm_clip_list.append(speaker_3dmm_clip[i*w: i*w + s_w])
                        speaker_audio2text_clip_list.append(speaker_audio2text_clip[i*w: i*w + s_w])
                        speaker_lip_clip_list.append(speaker_lip_clip[i*w: i*w + s_w])

                    motion_length = torch.tensor(motion_length_list)
                    speaker_audio_clip = torch.stack(speaker_audio_clip_list, dim=0)
                    speaker_emotion_clip = torch.stack(speaker_emotion_clip_list, dim=0)
                    speaker_3dmm_clip = torch.stack(speaker_3dmm_clip_list, dim=0)
                    speaker_audio2text_clip = torch.stack(speaker_audio2text_clip_list, dim=0)
                    speaker_lip_clip = torch.stack(speaker_lip_clip_list, dim=0)

                    motion_lengths.append(motion_length)
                    speaker_audios.append(speaker_audio_clip)
                    speaker_emotions.append(speaker_emotion_clip)
                    speaker_3dmms.append(speaker_3dmm_clip)
                    speaker_audio2text_clips_stack.append(speaker_audio2text_clip)
                    speaker_lip_clips_stack.append(speaker_lip_clip)


            motion_lengths = torch.cat(motion_lengths, dim=0)
            speaker_audios = torch.cat(speaker_audios, dim=0)
            speaker_emotions = torch.cat(speaker_emotions, dim=0)
            speaker_3dmms = torch.cat(speaker_3dmms, dim=0)
            speaker_audio2text_clips_stack = torch.cat(speaker_audio2text_clips_stack, dim=0)
            speaker_lip_clips_stack = torch.cat(speaker_lip_clips_stack, dim=0)

            sample_batch_size = torch.tensor(sample_batch_size)

            pred_listener_emotions = []
            pred_listener_emotions_mfg = []   
            all_batch_size = speaker_audios.shape[0]

            for i in range(math.ceil(all_batch_size / clip_batch_size)):
                speaker_audio_clip = speaker_audios[i * clip_batch_size: (i + 1) * clip_batch_size]
                speaker_emotion_clip = speaker_emotions[i * clip_batch_size: (i + 1) * clip_batch_size]
                speaker_3dmm_clip = speaker_3dmms[i * clip_batch_size: (i + 1) * clip_batch_size]
                motion_length = motion_lengths[i * clip_batch_size: (i + 1) * clip_batch_size]
                speaker_audio2text_clip = speaker_audio2text_clips_stack[i * clip_batch_size: (i + 1) * clip_batch_size]
                speaker_lip_clip = speaker_lip_clips_stack[i * clip_batch_size: (i + 1) * clip_batch_size]

                (speaker_audio_clip,
                 speaker_emotion_clip,
                 speaker_3dmm_clip,
                 speaker_audio2text_clip,
                 speaker_lip_clip) = (
                    speaker_audio_clip.to(self.device),
                    speaker_emotion_clip.to(self.device),
                    speaker_3dmm_clip.to(self.device),
                    speaker_audio2text_clip.to(self.device),
                    speaker_lip_clip.to(self.device)
                    )
            
                with torch.no_grad():
                    outputs = model(
                        speaker_audio_input=speaker_audio_clip,
                        speaker_emotion_input=speaker_emotion_clip,
                        speaker_3dmm_input=speaker_3dmm_clip,
                        motion_length=motion_length,
                        speaker_audio2text_input=speaker_audio2text_clip,
                        speaker_lip_input=speaker_lip_clip
                    )
                pred_listener_emotions.append(outputs["prediction_emotion"].detach().cpu())

                if self.use_mfg and self.mfg is not None:
                    with torch.no_grad():
                        mfg_out = self.mfg(
                            Z_f=speaker_emotion_clip,
                            Z_c=speaker_3dmm_clip,
                            L_text=speaker_audio2text_clip,
                            lengths=motion_length.to(self.device) if torch.is_tensor(motion_length) else None,
                            P_gt=None,
                            return_losses=False,
                            return_attn=True
                        )
                    P_mfg = mfg_out["P"]  # 期望 (B, N, l_w, D) or (N, B, l_w, D)
                    if P_mfg.dim() == 4 and P_mfg.shape[0] == self.mfg_num_preds:
                        P_mfg = P_mfg.permute(1, 0, 2, 3)  # (N,B,T,D)->(B,N,T,D)
                    pred_listener_emotions_mfg.append(P_mfg.detach().cpu())

                    sample_vals = {
                        "batch_idx": int(batch_idx),
                        "V_rel": mfg_out.get("V_rel", None).detach().cpu() if isinstance(mfg_out.get("V_rel", None), torch.Tensor) else None,
                        "V_subj": mfg_out.get("V_subj", None).detach().cpu() if isinstance(mfg_out.get("V_subj", None), torch.Tensor) else None,
                        "A_rel": mfg_out.get("A_rel", None).detach().cpu() if isinstance(mfg_out.get("A_rel", None), torch.Tensor) else None,
                        "A_subj": mfg_out.get("A_subj", None).detach().cpu() if isinstance(mfg_out.get("A_subj", None), torch.Tensor) else None,
                    }
                    test_attn_values.append(sample_vals)

            pred_listener_emotions = torch.cat(pred_listener_emotions, dim=0)  # (L', num_preds, l_w, 25)
            if len(pred_listener_emotions_mfg):
                pred_listener_emotions_mfg = torch.cat(pred_listener_emotions_mfg, dim=0)   # (L', N, l_w, D)

            pred_listener_emotion_list = []
            bounds = torch.cat((torch.tensor([0]), torch.cumsum(sample_batch_size, dim=0)), dim=0)
            intervals = list(zip(bounds[:-1], bounds[1:]))
            for (l, r) in intervals:
                pred_listener_emotion = pred_listener_emotions[l:r]  # (b', num_preds, l_w, 25)
                motion_length = motion_lengths[l:r]
                clip_length = torch.sum(motion_length, dim=0, keepdim=False)
                pred_listener_emotion = rearrange(pred_listener_emotion,
                                                  'b n w d -> n (b w) d')[:, :clip_length]

                if data_clamp:
                    pred_listener_emotion[:, :, :15] = torch.round(pred_listener_emotion[:, :, :15])

                pred_listener_emotion_list.append(pred_listener_emotion)
                pred_listener_emotions_all.extend(pred_listener_emotion_list)

                if self.use_mfg and self.mfg is not None and len(pred_listener_emotions_mfg):
                    pred_mfg = pred_listener_emotions_mfg[l:r]  # (b', N, l_w, D)
                    pred_mfg = rearrange(pred_mfg, 'b n w d -> n (b w) d')[:, :clip_length]
                    if data_clamp and pred_mfg.shape[-1] >= 15:
                        pred_mfg[:, :, :15] = torch.round(pred_mfg[:, :, :15])
                    pred_listener_emotions_all.append(pred_mfg)

   
        if len(pred_listener_emotions_all):
            GT_listener_emotions_all = post_processor.forward(
                prediction_list=pred_listener_emotions_all,
                target_list=GT_listener_emotions_all,)
      
        try:
            torch.save(test_attn_values, os.path.join(self.mfg_save_dir, "test_attn_values.pt"))
        except Exception as e:
            logger.debug(f"[MFG] save test attn values failed: {e}")

        try:
            torch.save({'GT': GT_listener_emotions_all, 'PRED': pred_listener_emotions_all},
                       f'results.pt')
            print("Successfully saved Tensor List")
        except Exception:
            print("Failed to save Tensor List")

        results = compute_metrics(
            input_speaker_emotions_all,
            pred_listener_emotions_all,
            GT_listener_emotions_all,
        )
        logger.info(results)
