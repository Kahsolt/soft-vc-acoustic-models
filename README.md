# soft-vc-acoustic-models

    使用 Soft-VC 制作应用时只需要重新训练自己的 Acoustic Model 就行了
    本仓库提供一些训练好的音色库，基础代码由官方soft-vc的四个仓库删减修改整合而来

----

### Voice Banks

| 音色 | 声库名(vbank) | 说明 | 语料 | 训练 |
| :-: | :-: | :-: | :-: | :-: |
| LJSpeech  | ljspeech  | 英语女性                 | LJSpeech公开数据集           | 32k steps |
| DataBaker | databaker | 汉语普通话女性            | DataBaker公开数据集          | 20k steps |
| 鏡音レン  | len        | 日语男性少年 (Vocaloid)  | 鏡音レン合成歌曲              | ? steps |
| 空詩音レミ| lemi       | 日语男性少年 (DeepVocal) | 空詩音レミ合成歌曲            | ? steps |
| はなinit  | hana       | 日语中性少年 (UTAU)      | はなinit合成歌曲+原始录音     | ? steps |
| 旭音エマ  | ema        | 日语中性少年 (UTAU)      | 旭音エマ合成歌曲              | ? steps |
| 狽音ウルシ| urushi     | 日语男性少年 (UTAU)      | 狽音ウルシ原始录音             | ? steps |
| 兰斯      | lansi      | 汉语普通话男性少年 (UTAU) | lansi2原始录音(有数据增强)    | ? steps |
| 钢琴      | piano      | 钢琴和弦乐               | 钢琴曲和少量弦乐协奏曲         | ? steps |
| 混杂      | mix        | 我擦我不好说             | 上述数据集的子集拼凑           | ? steps |

### Quick Start

⚪ Use pretrained voice banks

#### Commandline API

Download and put the pretrained checkpoint file at path `out\<vbank>\model-best.pt` where `vbank` is name of the voicebank

```cmd
python convert.py <vbank> <input>            => <input> can be both file or folder
python convert.py ljspeech test\000001.wav   => gen\000001_ljspeech.wav
python convert.py hana test                  => gen\*_hana.wav
```

converted outputs are in default generated under `gen` folder, named with suffix `<vbank>`

#### Programmatic API

```python
# imports
import torch
import torchaudio
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from scipy.io import wavfile

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load model
hubert   = torch.hub.load("bshall/hubert:main",          "hubert_soft").to(device)
acoustic = torch.hub.load("bshall/acoustic-model:main",  "hubert_soft").to(device)
hifigan  = torch.hub.load("bshall/hifigan:main", "hifigan_hubert_soft").to(device)

# load checkpoint
vbank = 'ljspeech'		         # or 'hana', etc..
ckpt_fp = f'out/{vbank}/model-best.pt'
ckpt = torch.load(ckpt_fp, map_location=device)
consume_prefix_in_state_dict_if_present(ckpt["acoustic-model"], "module.")
acoustic.load_state_dict(ckpt["acoustic-model"])

# load wavfile
wav_fp = r'test\000001.wav'	     # or whatever you want to convert from
source, sr = torchaudio.load(wav_fp)
source = torchaudio.functional.resample(source, sr, 16000)
source = source.unsqueeze(0).to(device)

# do soft-vc transform
with torch.inference_mode():
  units = hubert.units(source)
  mel = acoustic.generate(units).transpose(1, 2)
  target = hifigan(mel)

# save wavfile
y_hat = target.squeeze().cpu().numpy()
wavfile.write('converted.wav', 16000, y_hat)
```

see more details in `demo.ipynb` and `convert.py`

⚪ Train your own voice bank

Note that **each acoustic model** is typically treated as **one timbre**, so training on a multi-speaker dataset might probably get a confused timbre (just see the `mix` vbank). **Hence I will try to develop global-conditioned multi-timbre acoustic model in the near future :)**

1. prepare a folder containing \*.wav files (currently \*.mp3 not supported), aka. `wavpath`
2. make a config file `<vbank>.json` under `configs` folder where `vbank` is the name of your voice bank (refer to `configs\_template.json`)
3. install dependencies `pip install -r requirements.txt`
4. run bundled script `make_all.cmd <vbank> <wavpath>` (e.g. `make_all.cmd ljspeech C:\LJSpeech-1.1\wavs`) for full preprocess & train routine, then wait for 2000 years over :laughing:
  - if you got any error at midway, or just wants to operate step by step, refer to four steps in `Makefile`
  - `make dirs VBANK=<vbank> WAVPATH=<wavpath>` creates necessary folder hierachy and soft-links
  - `make units VBANK=<vbank>` encodes wavforms to hubert's hidden-units
  - `make mels VBANK=<vbank>` transforms wavforms to log-mel spectrograms 
  - `make train VBANK=<vbank>` trains the acoustic model with paired data (units, mels)

Note that preprocessed features are generated in `data\<vbank>\*`, while model checkpoints are saved in `out\<vbank>`

### References

Great thanks to the founding authors of Soft-VC! :lollipop:

```
@inproceedings{
  soft-vc-2022,
  author={van Niekerk, Benjamin and Carbonneau, Marc-André and Zaïdi, Julian and Baas, Matthew and Seuté, Hugo and Kamper, Herman},
  booktitle={ICASSP}, 
  title={A Comparison of Discrete and Soft Speech Units for Improved Voice Conversion}, 
  year={2022}
}
```

- soft-vc paper: [https://ieeexplore.ieee.org/abstract/document/9746484](https://ieeexplore.ieee.org/abstract/document/9746484)
- soft-vc code: [https://github.com/bshall/soft-vc](https://github.com/bshall/soft-vc)
  - hubert: [https://github.com/bshall/hubert](https://github.com/bshall/hubert)
  - acoustic-model: [https://github.com/bshall/acoustic-model](https://github.com/bshall/acoustic-model)
  - hifigan: [https://github.com/bshall/hifigan](https://github.com/bshall/hifigan)

----

by Armit
2022/09/12 
