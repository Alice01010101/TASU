import torch
from torch import nn
import torchvision
import pdb

import os
import numpy as np
import torch

import laion_clap
from .sqrtm import sqrtm
from .pann import Cnn14_16k, Cnn14

class VisualNet(nn.Module):
    def __init__(self, use_rgb, use_depth, use_audio):
        super(VisualNet, self).__init__()
        assert use_rgb or use_depth or use_audio
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.use_audio = use_audio

        in_channel = use_rgb * 3 + use_depth + use_audio
        conv1 = nn.Conv2d(in_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        layers = list(torchvision.models.resnet18(pretrained=True).children())[1:-1]
        self.feature_extraction = nn.Sequential(conv1, *layers)  # features before conv1x1
        self.predictor = nn.Sequential(nn.Linear(512, 1))

    def forward(self, inputs):
        audio_feature = self.feature_extraction(inputs).squeeze(-1).squeeze(-1)
        pred = self.predictor(audio_feature)

        return pred


# class SI_SNR(nn.Module):
#     def __init__(self):
#         super(SI_SNR, self).__init__()
#         self.EPS = 1e-8

#     def forward(self, source, estimate_source):

#         if source.shape[-1] > estimate_source.shape[-1]:
#             source = source[..., :estimate_source.shape[-1]]
#         if source.shape[-1] < estimate_source.shape[-1]:
#             estimate_source = estimate_source[..., :source.shape[-1]]

#         # step 1: Zero-mean norm
#         source = source - torch.mean(source, dim=-1, keepdim=True)
#         estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

#         # step 2: Cal si_snr
#         # s_target = <s', s>s / ||s||^2
#         ref_energy = torch.sum(source ** 2, dim = -1, keepdim=True) + self.EPS
#         proj = torch.sum(source * estimate_source, dim = -1, keepdim=True) * source / ref_energy
#         # e_noise = s' - s_target
#         noise = estimate_source - proj
#         # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
#         ratio = torch.sum(proj ** 2, dim = -1) / (torch.sum(noise ** 2, dim = -1) + self.EPS)
#         sisnr = 10 * torch.log10(ratio + self.EPS)

#         sisnr = torch.mean(sisnr)

#         return sisnr
    

def sisdr(source, estimate_source):
        EPS = 1e-8
        if source.shape[-1] > estimate_source.shape[-1]:
            source = source[..., :estimate_source.shape[-1]]
        if source.shape[-1] < estimate_source.shape[-1]:
            estimate_source = estimate_source[..., :source.shape[-1]]

        # step 1: Zero-mean norm
        source = source - torch.mean(source, dim=-1, keepdim=True)
        estimate_source = estimate_source - torch.mean(estimate_source, dim=-1, keepdim=True)

        # step 2: Cal si_snr
        # s_target = <s', s>s / ||s||^2
        ref_energy = torch.sum(source ** 2, dim = -1, keepdim=True) + EPS
        proj = torch.sum(source * estimate_source, dim = -1, keepdim=True) * source / ref_energy
        # e_noise = s' - s_target
        noise = estimate_source - proj
        # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2)
        ratio = torch.sum(proj ** 2, dim = -1) / (torch.sum(noise ** 2, dim = -1) + EPS)
        sisnr = 10 * torch.log10(ratio + EPS)

        sisnr = torch.mean(sisnr)

        return sisnr


class RTE(nn.Module):
    def __init__(self):
        super(RTE,self).__init__()
        self.estimator = VisualNet(use_rgb=False, use_depth=False, use_audio=True)
        pretrained_weights = 'data/rt60_estimator.pth'
        self.estimator.load_state_dict(torch.load(pretrained_weights, map_location='cpu')['predictor'])

    def forward(self,x):
        self.estimator = self.estimator.to(x.device).eval()
        stft = torch.stft(x, n_fft=512, hop_length=160, win_length=400, window=torch.hamming_window(400, device=x.device),
                          pad_mode='constant', return_complex=True)
        spec = torch.log1p(stft.abs()).unsqueeze(1)
        with torch.no_grad():
            estimated_rt60 = self.estimator(spec)
        return estimated_rt60


class FrechetAudioDistance(nn.Module):
    def __init__(self):
        super(FrechetAudioDistance, self).__init__()
        #self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        self.model = torch.hub.load(repo_or_dir='/home/pantianrui/.cache/torch/hub/harritaylor_torchvggish_master', source='local', model='vggish')

    def calculate_embd_statistics(self, embd):
        mu = torch.mean(embd, axis=0) # 512
        sigma = torch.cov(embd.T) # ([512,512])
        return mu, sigma

    def calculate_frechet_distance(self, mu_gen, sigma_gen, mu_tar, sigma_tar):
        mu_diff = mu_gen - mu_tar
        offset = torch.eye(sigma_gen.shape[0]) * 1e-6
        offset = offset.to(mu_gen.device)
        fad_score = mu_diff.dot(mu_diff) + torch.trace(sigma_tar) + torch.trace(sigma_gen) - 2 * torch.trace(torch.real(sqrtm(torch.matmul(sigma_gen+offset, sigma_tar+offset), method='maji')))
        return fad_score

    def forward(self, generate_wav, target_wav):
        self.model.postprocess = False
        self.model.embeddings = torch.nn.Sequential(*list(self.model.embeddings.children())[:-1])
        self.model.device = generate_wav.device

        generate_embd = []
        target_embd = []
        for i in range(generate_wav.shape[0]):
            temp_a = self.model.forward(generate_wav[i].cpu().numpy(),16000)
            temp_b = self.model.forward(target_wav[i].cpu().numpy(),16000)
            generate_embd.append(temp_a)
            target_embd.append(temp_b)
        generate_embd = torch.cat(generate_embd, dim=0)
        target_embd = torch.cat(target_embd, dim=0)

        # generate_embd = torch.from_numpy(self.clap_model.get_audio_embedding_from_data(x=generate_wav.cpu().numpy())) #(B,512)
        # target_embd = torch.from_numpy(self.clap_model.get_audio_embedding_from_data(x=target_wav.cpu().numpy()))

        mu_gen, sigma_gen = self.calculate_embd_statistics(generate_embd)
        mu_tar, sigma_tar = self.calculate_embd_statistics(target_embd)

        fad_score = self.calculate_frechet_distance(mu_gen, sigma_gen, mu_tar, sigma_tar)

        return fad_score

class FrechetDistance(nn.Module):
    def __init__(self):
        super(FrechetDistance, self).__init__()
        #self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        #self.model = torch.hub.load(repo_or_dir='/home/pantianrui/.cache/torch/hub/harritaylor_torchvggish_master', source='local', model='vggish')
        self.model = Cnn14_16k(
                    sample_rate=16000,
                    window_size=512,
                    hop_size=160,
                    mel_bins=64,
                    fmin=50,
                    fmax=8000,
                    classes_num=527
                )
        checkpoint = torch.load('/home/pantianrui/av_ldm/data/Cnn14_16k_mAP=0.438.pth', map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])


    def calculate_embd_statistics(self, embd):
        mu = torch.mean(embd, axis=0) # 512
        sigma = torch.cov(embd.T) # ([512,512])
        return mu, sigma

    # def calculate_frechet_distance(self, mu_gen, sigma_gen, mu_tar, sigma_tar):
    #     mu_diff = mu_tar - mu_gen
    #     fd_score = mu_diff.dot(mu_diff) + torch.trace(sigma_tar) + torch.trace(sigma_gen) - 2 * torch.trace(torch.real(sqrtm(torch.matmul(sigma_tar, sigma_gen), method='maji')))
    #     return fd_score

    def calculate_frechet_distance(self, mu_gen, sigma_gen, mu_tar, sigma_tar):
        mu_diff = mu_gen - mu_tar
        offset = torch.eye(sigma_gen.shape[0]) * 1e-6
        offset = offset.to(mu_gen.device)
        fd_score = mu_diff.dot(mu_diff) + torch.trace(sigma_tar) + torch.trace(sigma_gen) - 2 * torch.trace(torch.real(sqrtm(torch.matmul(sigma_gen+offset, sigma_tar+offset), method='maji')))
        return fd_score

    def forward(self, generate_wav, target_wav):

        self.model = self.model.to(generate_wav.device).eval()
        generate_embd = []
        target_embd = []
        for i in range(generate_wav.shape[0]): 
            temp_a = self.model(generate_wav[i].unsqueeze(0),None)
            temp_b = self.model(target_wav[i].unsqueeze(0),None)
            generate_embd.append(temp_a['embedding'].data[0].unsqueeze(0))
            target_embd.append(temp_b['embedding'].data[0].unsqueeze(0))
        generate_embd = torch.cat(generate_embd, dim=0)
        target_embd = torch.cat(target_embd, dim=0)

        # generate_embd = torch.from_numpy(self.clap_model.get_audio_embedding_from_data(x=generate_wav.cpu().numpy())) #(B,512)
        # target_embd = torch.from_numpy(self.clap_model.get_audio_embedding_from_data(x=target_wav.cpu().numpy()))

        mu_gen, sigma_gen = self.calculate_embd_statistics(generate_embd)
        mu_tar, sigma_tar = self.calculate_embd_statistics(target_embd)

        fd_score = self.calculate_frechet_distance(mu_gen, sigma_gen, mu_tar, sigma_tar)

        return fd_score

class InceptionScore(nn.Module):
    def __init__(self):
        super(InceptionScore, self).__init__()
        features_list = ["2048", "logits"]
        self.mel_model = Cnn14(
            #features_list=features_list,
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )
        checkpoint = torch.load('/home/pantianrui/av_ldm/data/Cnn14_16k_mAP=0.438.pth')
        self.mel_model.load_state_dict(checkpoint['model'])

    def calculate_isc(self, featuresdict, feat_layer_name, rng_seed, samples_shuffle, splits):
        print("Computing Inception Score")

        features = featuresdict[feat_layer_name]

        assert torch.is_tensor(features) and features.dim() == 2
        N, C = features.shape
        if samples_shuffle:
            rng = np.random.RandomState(rng_seed)
            features = features[rng.permutation(N), :]
        features = features.double()

        p = features.softmax(dim=1)
        log_p = features.log_softmax(dim=1)

        scores = []
        for i in range(splits):
            p_chunk = p[(i * N // splits) : ((i + 1) * N // splits), :]  # 一部分的预测概率
            log_p_chunk = log_p[(i * N // splits) : ((i + 1) * N // splits), :]  # log
            q_chunk = p_chunk.mean(dim=0, keepdim=True)  # 概率的均值
            kl = p_chunk * (log_p_chunk - q_chunk.log())  #
            kl = kl.sum(dim=1).mean().exp().item()
            scores.append(kl)
        
        pdb.set_trace()
        # print("scores",scores)
        return {
            "inception_score_mean": float(np.mean(scores)),
            "inception_score_std": float(np.std(scores)),
        }

    def get_featuresdict(self, wavs):
        out = None
        out_meta = {}

        # transforms=StandardNormalizeAudio()
        for i in range(wavs.shape[0]):
            waveform = wavs[i].unsqueeze(0)
            with torch.no_grad():
                featuresdict = self.mel_model(waveform) # "logits": [1, 527]

                featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

            if out is None:
                out = featuresdict
            else:
                out = {k: out[k] + featuresdict[k] for k in out.keys()}

        out = {k: torch.cat(v, dim=0) for k, v in out.items()}
        return {**out, **out_meta}

    def forward(self, generate_wav, target_wav):
        self.mel_model.to(generate_wav.device)
        self.mel_model.eval()

        generate_features = self.get_featuresdict(generate_wav)
        #target_features = self.get_featuresdict(target_wav)

        # metric_kl, kl_ref, paths_1 = calculate_kl(
        #     generate_features, target_features, "logits", same_name
        # )

        #pdb.set_trace()
        metric_isc = self.calculate_isc(
            generate_features,
            feat_layer_name="logits",
            splits=1,
            samples_shuffle=True,
            rng_seed=2020,
        )

        return generate_features


class KLScore(nn.Module):
    def __init__(self):
        super(KLScore, self).__init__()
        self.mel_model = Cnn14(
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )
        checkpoint = torch.load('/home/pantianrui/av_ldm/data/Cnn14_16k_mAP=0.438.pth')
        self.mel_model.load_state_dict(checkpoint['model'])

    def get_featuresdict(self, wavs):
        out = None
        out_meta = {}

        # transforms=StandardNormalizeAudio()
        for i in range(wavs.shape[0]):
            waveform = wavs[i].unsqueeze(0)
            with torch.no_grad():
                featuresdict = self.mel_model(waveform) # "logits": [1, 527]

                featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

            if out is None:
                out = featuresdict
            else:
                out = {k: out[k] + featuresdict[k] for k in out.keys()}

        out = {k: torch.cat(v, dim=0) for k, v in out.items()}
        return {**out, **out_meta}

    def calculate_kl(self, generate_featuresdict, target_featuresdict):
        generate_features = generate_featuresdict['logits']
        target_features = target_featuresdict['logits']
        EPS = 1e-6

        # kl_ref = torch.nn.functional.kl_div(
        #     (generate_features.softmax(dim=1) + EPS).log(),
        #     target_features.softmax(dim=1),
        #     reduction="none",
        # ) / len(generate_features)
        # kl_ref = torch.mean(kl_ref, dim=-1)

        kl_softmax = torch.nn.functional.kl_div(
            (generate_features.softmax(dim=1) + EPS).log(),
            target_features.softmax(dim=1),
            reduction="sum",
        ) / len(generate_features)

        # For multi-class audio clips, this formulation could be better
        kl_sigmoid = torch.nn.functional.kl_div(
            (generate_features.sigmoid() + EPS).log(), target_features.sigmoid(), reduction="sum"
        ) / len(generate_features)

        return {
            "kullback_leibler_divergence_sigmoid": float(kl_sigmoid),
            "kullback_leibler_divergence_softmax": float(kl_softmax),
        }

    def forward(self, generate_wav, target_wav):
        self.mel_model.to(generate_wav.device)
        self.mel_model.eval()

        # generate_featuresdict = self.get_featuresdict(generate_wav)
        # target_featuresdict = self.get_featuresdict(target_wav)
        generate_featuresdict = self.mel_model(generate_wav)
        target_featuresdict = self.mel_model(target_wav)

        metric_kl = self.calculate_kl(
            generate_featuresdict, target_featuresdict
        )

        return metric_kl["kullback_leibler_divergence_softmax"], metric_kl["kullback_leibler_divergence_sigmoid"]
