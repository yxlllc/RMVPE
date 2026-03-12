import argparse
from typing import Dict, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel

from src import E2E0
from src.constants import *

import torch


class MelSpectrogram_ONNX(nn.Module):
    def __init__(
            self,
            n_mel_channels,
            sampling_rate,
            win_length,
            hop_length,
            n_fft=None,
            mel_fmin=0,
            mel_fmax=None,
            clamp=1e-5
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        mel_basis = mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = win_length if n_fft is None else n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp

    def forward(self, audio, center=True):
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=audio.device),
            center=center,
            return_complex=False
        )
        magnitude = torch.sqrt(torch.sum(fft ** 2, dim=-1))
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE_ONNX(nn.Module):
    def __init__(self, hop_length):
        super().__init__()
        self.hop_length = hop_length
        self.seg_length = 32 * hop_length
        self.model = E2E0(4, 1, (2, 2))
        self.mel_extractor = MelSpectrogram_ONNX(
            N_MELS, SAMPLE_RATE, WINDOW_LENGTH, hop_length, None, MEL_FMIN, MEL_FMAX
        )

    # noinspection PyMethodMayBeStatic
    def decode(self, hidden, threshold=0.03):
        idx = torch.arange(N_CLASS, device=hidden.device)[None, None, :]  # [B=1, T=1, N]
        idx_cents = idx * 20 + CONST  # [B=1, N]
        center = torch.argmax(hidden, dim=2, keepdim=True)  # [B, T, 1]
        start = torch.clip(center - 4, min=0)  # [B, T, 1]
        end = torch.clip(center + 5, max=N_CLASS)  # [B, T, 1]
        idx_mask = (idx >= start) & (idx < end)  # [B, T, N]
        weights = hidden * idx_mask  # [B, T, N]
        product_sum = torch.sum(weights * idx_cents, dim=2)  # [B, T]
        weight_sum = torch.sum(weights, dim=2)  # [B, T]
        cents = product_sum / (weight_sum + (weight_sum == 0))  # avoid dividing by zero, [B, T]
        f0 = 10 * 2 ** (cents / 1200)
        uv = hidden.max(dim=2)[0] < threshold  # [B, T]
        return f0 * ~uv, uv

    def forward(self, waveform, threshold):
        B, T = waveform.shape
        n_frames = T // self.hop_length + 1 
        T1 = T + self.hop_length
        T_pad = self.seg_length * ((T1 - 1) // self.seg_length + 1) - T1
        waveform = F.pad(waveform, (0, T_pad))
        mel = self.mel_extractor(waveform, center=True)
        hidden = self.model(mel)
        f0, uv = self.decode(hidden[:, :n_frames], threshold=threshold)
        return f0, uv


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="path to the output onnx file",
    )
    parser.add_argument(
        "-hop",
        "--hop_length",
        type=str,
        required=False,
        default=160,
        help="hop_length under 16khz sampling rate | default: 160",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="whether to optimize the generated ONNX graph"
    )
    return parser.parse_args(args=args, namespace=namespace)


def onnx_override_io_shapes(
        model,  # ModelProto
        input_shapes: Dict[str, Tuple[Union[str, int]]] = None,
        output_shapes: Dict[str, Tuple[Union[str, int]]] = None,
):
    """
    Override the shapes of inputs/outputs of the model graph (in-place operation).
    :param model: model to perform the operation on
    :param input_shapes: a dict with keys as input/output names and values as shape tuples
    :param output_shapes: the same as input_shapes
    """
    def _override_shapes(
            shape_list_old,  # RepeatedCompositeFieldContainer[ValueInfoProto]
            shape_dict_new: Dict[str, Tuple[Union[str, int]]]):
        for value_info in shape_list_old:
            if value_info.name in shape_dict_new:
                name = value_info.name
                dims = value_info.type.tensor_type.shape.dim
                assert len(shape_dict_new[name]) == len(dims), \
                    f'Number of given and existing dimensions mismatch: {name}'
                for i, dim in enumerate(shape_dict_new[name]):
                    if isinstance(dim, int):
                        dims[i].dim_param = ''
                        dims[i].dim_value = dim
                    else:
                        dims[i].dim_value = 0
                        dims[i].dim_param = dim

    if input_shapes is not None:
        _override_shapes(model.graph.input, input_shapes)
    if output_shapes is not None:
        _override_shapes(model.graph.output, output_shapes)


def export():
    cmd = parse_args()

    model_path = cmd.model
    output_path = cmd.output

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('loading model and audio')
    rmvpe = RMVPE_ONNX(hop_length=cmd.hop_length)
    rmvpe.model.load_state_dict(torch.load(model_path)['model'])
    rmvpe.eval().to(device)
    waveform = torch.randn(2, 114514, dtype=torch.float32, device=device).clip(min=-1., max=1.)
    threshold = torch.tensor(0.03, dtype=torch.float32, device=device)
    print('start exporting ...')
    with torch.no_grad():
        torch.onnx.export(
            rmvpe,
            (
                waveform,
                threshold,
            ),
            output_path,
            input_names=[
                'waveform',
                'threshold'
            ],
            output_names=[
                'f0',
                'uv'
            ],
            dynamic_axes={
                'waveform': {
                    0: 'batch_size',
                    1: 'n_samples'
                },
                'f0': {
                    0: 'batch_size',
                    1: 'n_frames'
                },
                'uv': {
                    0: 'batch_size',
                    1: 'n_frames'
                }
            },
            opset_version=17
        )
    if cmd.optimize:
        import onnx
        import onnxsim
        print('start optimizing ...')
        model = onnx.load(output_path)
        model, check = onnxsim.simplify(
            model,
            include_subgraph=True
        )
        assert check, 'Simplified ONNX model could not be validated'
        onnx.save(model, output_path)


if __name__ == '__main__':
    export()
