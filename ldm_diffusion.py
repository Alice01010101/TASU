from audio_diffusion_pytorch import DiffusionModel, UNetV0, VDiffusion, VSampler, StyleVDiffusion, StyleVSampler
from metrics.criterion import sisdr, RTE, FrechetDistance, FrechetAudioDistance
import torch.nn as nn
import pytorch_lightning as pl
from Amphion.models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
import torch.nn.functional as F
import speechmetrics
import os
import torch
import torchvision
import laion_clap
import pdb

class FACodecDiffusion(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.save_hyperparameters(args)
        
        num_channel = 4
        conv1 = nn.Conv2d(num_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layers = list(torchvision.models.resnet18(pretrained=True).children())[1:-2]
        #self.cnn = nn.Sequential(conv1, *layers)

        self.model = DiffusionModel(
            net_t=UNetV0, # The model type used for diffusion (U-Net V0 in this case)
            dim=2, # for spectrogram we use 2D-CNN
            in_channels=2, # U-Net: number of input (audio) channels
            out_channels=1, # U-Net: number of output (audio) channels
            channels=[256, 512, 768, 1280, 1280], # U-Net: channels at each layer
            factors=[2, 2, 2, 2, 1], # U-Net: downsampling and upsampling factors at each layer
            items=[2, 2, 2, 2, 2], # U-Net: number of repeating items at each layer
            attentions=[0, 0, 1, 1, 1], # U-Net: attention enabled/disabled at each layer
            attention_heads=8, # U-Net: number of attention heads per attention item
            attention_features=64, # U-Net: number of attention features per attention item
            diffusion_t=StyleVDiffusion, # The diffusion method used
            sampler_t=StyleVSampler, # The diffusion sampler used
            # use_embedding_cfg=True, # Use classifier free guidance
            # embedding_max_length=1, # Maximum length of the embeddings
            embedding_features=512, # U-Net: embedding features
            cross_attentions=[0, 0, 1, 1, 1], # U-Net: cross-attention enabled/disabled at each layer 
        )
        self.fa_encoder = FACodecEncoder(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )
        self.fa_decoder = FACodecDecoder(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
        )
        # encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder.bin")
        # decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder.bin")

        encoder_ckpt ='/home/pantianrui/.cache/huggingface/hub/models--amphion--naturalspeech3_facodec/snapshots/314afc3ea1455ba881a0e484ef9408b6cb996736/ns3_facodec_encoder.bin'
        decoder_ckpt = '/home/pantianrui/.cache/huggingface/hub/models--amphion--naturalspeech3_facodec/snapshots/314afc3ea1455ba881a0e484ef9408b6cb996736/ns3_facodec_decoder.bin'

        self.fa_encoder.load_state_dict(torch.load(encoder_ckpt))
        self.fa_decoder.load_state_dict(torch.load(decoder_ckpt))

        self.fa_encoder.eval()
        self.fa_decoder.eval()

        self.rte_estimator = RTE()
        # self.mosnet = speechmetrics.load('mosnet',window=None)
        self.pesq_eval = PerceptualEvaluationSpeechQuality(16000,'wb')
        self.fd_score_eval = FrechetDistance()

        self.best_val_loss = 1000
        self.args = args

        self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')

    def configure_optimizers(self):
        return torch.optim.AdamW(params=list(self.model.parameters()), lr=1e-4, betas= (0.95, 0.999), eps=1e-6, weight_decay=1e-3)

    def get_visual_feat(self, batch):
        visual_input = []
        visual_input.append(batch['depth'])
        visual_input.append(batch['rgb'])
        visual_input = torch.cat(visual_input, dim=1)

        self.cnn = self.cnn.to(visual_input.device)
        self.cnn = self.cnn.to(dtype = visual_input.dtype)
        img_feat = self.cnn(visual_input)
        img_feat = img_feat.reshape(img_feat.size(0), img_feat.size(1), -1).permute(0, 2, 1) #(B,108,512)

        return img_feat

    def training_step(self, batch, batch_idx):
        stats = {'loss': 0}
        self.model.train()
        audio,target,cond = batch['src_wav'], batch['recv_wav'], batch['recv_wav'] #dtype:torch.float32
        
        audio = audio.unsqueeze(1)
        target = target.unsqueeze(1)
        with torch.no_grad():

            ################################################################################################################################
            # img_feat = self.get_visual_feat(batch) #(B,108,512)
            # img_feat = img_feat.to(torch.float32)
            cond_embed = torch.from_numpy(self.clap_model.get_audio_embedding_from_data(x=cond.cpu().numpy())).unsqueeze(1).to(audio.device)
            ################################################################################################################################

            #encode
            audio_encode = self.fa_encoder(audio) #(B,1,40960) -> (B,256,205)
            target_encode = self.fa_encoder(target)
            #quantize
            audio_vq_post_emb, audio_vq_id, _, audio_quantized, audio_spk_embs = self.fa_decoder(audio_encode, eval_vq=False, vq=True)  #(B,256,205)
            target_vq_post_emb, target_vq_id, _, target_quantized, target_spk_embs = self.fa_decoder(target_encode, eval_vq=False, vq=True)  #target_vq_post_emb.shape (12, 256, 208) #target_vq_id.shape (6,12,208)

        stats['loss'] += self.model(target_vq_post_emb.unsqueeze(1), audio_vq_post_emb.unsqueeze(1), embedding=cond_embed)
        #stats['loss'] += self.model(target_vq_post_emb.unsqueeze(1), audio_vq_post_emb.unsqueeze(1))
        return stats

    def training_epoch_end(self, outputs):
        metrics = outputs[0].keys()
        output_str = f'Train epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.Tensor([output[metric] for output in outputs]).mean()
            #self.logger.experiment.add_scalar(f'{metric}', avg_value, self.current_epoch)
            self.log(f'{metric}', avg_value, on_step=False, on_epoch=True, sync_dist=True)
            output_str += f'{metric}: {avg_value:.4f}, '

        self.print(output_str[:-2])

    def eval_stft(self,wav):
        spec = torch.stft(wav, n_fft=512, hop_length=160, win_length=400,
                        window=torch.hamming_window(400, device=wav.device), pad_mode='constant',
                        return_complex=False)

        return spec

    def validation_step(self, batch, batch_idx, test=False):
        # Evaluate on valid set
        self.model.eval()
        states = {}
        test_audio, test_target, test_cond = batch['src_wav'], batch['recv_wav'], batch['recv_wav']
        test_audio = test_audio.unsqueeze(1)
        test_target = test_target.unsqueeze(1)
        with torch.no_grad():
            ##############################for condition embedding############################################################################################
            # img_feat = self.get_visual_feat(batch) #(B,108,512)
            # img_feat = img_feat.to(torch.float32)
            test_cond_embed = torch.from_numpy(self.clap_model.get_audio_embedding_from_data(x=test_cond.cpu().numpy())).unsqueeze(1).to(test_audio.device)
            #################################################################################################################################################
            #encode
            test_audio_encode = self.fa_encoder(test_audio) #(B,1,40960) -> (B,256,205)
            test_target_encode = self.fa_encoder(test_target)
            #quantize
            test_audio_vq_post_emb, test_audio_vq_id, _, test_audio_quantized, test_audio_spk_embs = self.fa_decoder(test_audio_encode, eval_vq=False, vq=True)  #(B,256,205)
            test_target_vq_post_emb, test_target_vq_id, _, test_target_quantized, test_target_spk_embs = self.fa_decoder(test_target_encode, eval_vq=False, vq=True) 

            test_loss = self.model(test_target_vq_post_emb.unsqueeze(1), test_audio_vq_post_emb.unsqueeze(1), embedding=test_cond_embed).item()
            #test_loss = self.model(test_target_vq_post_emb.unsqueeze(1), test_audio_vq_post_emb.unsqueeze(1)).item()


            ##################################################################################################################################################
            noise = torch.randn(test_audio.shape[0], 1, 256, 208, device=test_audio.device)
            
            diffusion_sample = self.model.sample(noise, test_audio_vq_post_emb.unsqueeze(1), embedding=test_cond_embed, num_steps=200, show_progress=True)
            #diffusion_sample = self.model.sample(noise, test_audio_vq_post_emb.unsqueeze(1), num_steps=200, show_progress=True)
            
            generate_wav = self.fa_decoder.inference(diffusion_sample.squeeze(1),test_target_spk_embs)
            target_wav = self.fa_decoder.inference(test_target_vq_post_emb, test_target_spk_embs)


            generate_wav = generate_wav.squeeze(1)
            target_wav = target_wav.squeeze(1)
            gen_rte = self.rte_estimator(generate_wav)
            target_rte = self.rte_estimator(target_wav) 
            rte = (gen_rte - target_rte).abs().mean()

            # mosnet = speechmetrics.load('mosnet',window=None)
            # mos_score = mosnet(generate_wav.cpu(), target_wav.cpu(), rate=16000)['mosnet'].item()
            # mos_target_score = mosnet(target_wav.cpu(), generate_wav.cpu(), rate=16000)['mosnet'].item()
            # mose = abs(mos_score - mos_target_score)

            #stft_distance = F.mse_loss(self.eval_stft(generate_wav), self.eval_stft(target_wav))
            fd_score = self.fd_score_eval(generate_wav, target_wav)
            pesq = self.pesq_eval(generate_wav, target_wav)

            # fad_score_eval = FrechetAudioDistance()
            # fad_score = fad_score_eval(generate_wav, target_wav)
            ####################################################################################################

        states['valid_loss'] = test_loss
        states['speech_rt60'] = rte
        #states['mose'] = mose
        #states['stfte'] = stft_distance
        #states['fad_score'] = fad_score
        states['fd_score'] = fd_score
        states['pesq'] = pesq
        
        return states

    
    def validation_epoch_end(self, outputs):
        gathered_outputs = self.all_gather(outputs)
        metrics = gathered_outputs[0].keys()
        output_str = f'Val epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.stack([output[metric] for output in gathered_outputs], dim=0).mean()
            self.log(f'{metric}', avg_value, on_step=False, on_epoch=True, sync_dist=True)
            #self.logger.experiment.add_scalar(f'{metric}', avg_value, self.current_epoch)
            output_str += f'{metric}: {avg_value:.4f}, '

        self.print(output_str[:-2])

    def test_step(self, batch, batch_idx, test=False):
        states = self.validation_step(batch, batch_idx, test=False)

        return states

    def test_epoch_end(self, outputs):
        gathered_outputs = self.all_gather(outputs)
        metrics = gathered_outputs[0].keys()
        output_str = f'Test epoch {self.current_epoch}, '
        for metric in metrics:
            avg_value = torch.stack([output[metric] for output in gathered_outputs], dim=0).mean()
            self.log(f'{metric}', avg_value, on_step=False, on_epoch=True, sync_dist=True)
            #self.logger.experiment.add_scalar(f'{metric}', avg_value, self.current_epoch)
            output_str += f'{metric}: {avg_value:.4f}, '

        self.print(output_str[:-2])
