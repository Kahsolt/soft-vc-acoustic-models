# soft-vc-acoustic-models

    Provides scripts for conveniently training your own acoustic model for Soft-VC, also providing a list of experimentally pretrained acoustic models.

----

使用 [Soft-VC](https://github.com/bshall/soft-vc) 制作变声器应用时只需要重新训练自己的声学模型 [Acoustic Model](https://github.com/bshall/acoustic-model) 就行了  
本仓库提供一些整合脚本可以很方便地训练自己的声学模型 :)，你也可以尝试下载一些预先训练好的音色库 (虽然几乎都很垃圾不好用！！)  
基础代码由官方soft-vc的四个仓库删减修改整合而来  


### Voice Banks

| 音色 | 声库名(vbank) | 说明 | 语料 | 语料时长 | 最优检查点 | 听感状态 |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| LJSpeech  | ljspeech  | 英语女性成人             | LJSpeech公开数据集            | 24h | 32k steps | 可用 |
| DataBaker | databaker | 汉语普通话女性成人        | DataBaker公开数据集           | 10h | 25k steps | 可用 |
| 阿        | aak        | 日语男性少年             | 游戏内语音(明日方舟)          |  | ? steps |  |
| 卡达      | click      | 日语女性少年             | 游戏内语音(明日方舟)          |  | ? steps |  |
| 红云      | vermeil    | 日语女性少年             | 游戏内语音(明日方舟)          |  | ? steps |  |
| 空(旅行者) | aether    | 日语男性少年             | 游戏内语音(原神)              |  | ? steps |  |
| 派蒙      | paimon     | 日语女性幼儿             | 游戏内语音(原神)              |  | ? steps |  |
| 爽        | sou       | 日语男性少年              | 歌声提取（空詩音レミ的中之人） | 0.243h | 11k steps | 撕裂，局部平声 |
| 空詩音レミ | lemi      | 日语男性少年 (DeepVocal) | 歌声合成导出                  | 0.351h | 34k steps |  |
| 鏡音レン   | len       | 日语男性少年 (Vocaloid)  | 歌声合成导出                  | 0.575h | 36k steps |  |
| はなinit  | hana       | 日语中性少年 (UTAU)      | 歌声合成导出+声库录音         | 1.672h | 37k steps |  |
| 旭音エマ   | ema       | 日语中性少年 (UTAU)      | 歌声合成导出+声库录音         | 0.433h | 2k steps |  |
| 狽音ウルシ | urushi    | 日语男性少年 (UTAU)       | 声库录音                    | 0.190h | 36k steps | 完全平声 |
| 兰斯      | lansi      | 汉语普通话男性少年 (UTAU) | 声库录音(+数据增强)          | 5.417h | 21k steps | 完全平声 |
| 钢琴      | piano      | 钢琴和弦乐               | 钢琴曲和少量弦乐协奏曲        | 0.800h | 32K steps |  |

⚠️ **自然人声音受到当地法律保护，应仅出于个人学习、艺术欣赏、课堂教学或者科学研究等目的作必要使用。**  
⚠️ **The voice of natural persons is protected by local laws and shall be used ONLY for necessary purposes such as personal study, artistic appreciation, teaching or scientific research.**  

Model checkpoints could be downloaded from here: [TODO: upload to cloud disk](http://no.where.to.go).  
We equally train each vbank for `40k` steps, but only save the best checkpoint.

ℹ️ **Note: not all vbanks are pleasing due to very very limited training data**, please check the audio samples in `index.html` for a comprehensive understanding.  
For discussions on how many data is necessarily needed to train a satisfactory voice bank, refer to this repo: [soft-vc-acoustic-model-ablation-study](https://github.com/Kahsolt/soft-vc-acoustic-model-ablation-study)


### Quick Start

⚪ Use pretrained voice banks

#### Commandline API

Download and put the pretrained checkpoint file at path `out\<vbank>\model-best.pt` where `vbank` is name of the voicebank

```cmd
python infer.py <vbank> <input>            => <input> can be both file or folder
python infer.py ljspeech test\000001.wav   => gen\000001_ljspeech.wav
python infer.py hana test                  => gen\*_hana.wav
```

converted outputs are in default generated under `gen` folder, files named with tailing suffix `_<vbank>`

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
vbank = 'ljspeech'		           # or 'hana', etc..
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

see more details in `demo.ipynb` and `infer.py`

⚪ Train your own voice bank

ℹ️ Note that **each acoustic model** is typically treated as **one timbre**, so training on a multi-speaker dataset might probably get a confused timbre. **Hence I will probably try to develop global-conditioned multi-timbre acoustic model in the near future :)**

1. prepare a folder containing \*.wav files (currently \*.mp3 not supported), aka. `wavpath`
2. (optional) create a config file `<config>.json` under `configs` folder (refer to `configs\default.json` which is defaultly used)
3. install dependencies `pip install -r requirements.txt`
4. use the two-stage separated scripts for preprocessing and training routine:
  - preprocess with `make_preprocess.cmd <vbank> <wavpath>`
    - e.g. `make_preprocess.cmd ljspeech C:\LJSpeech-1.1\wavs`
  - train with `make_train.cmd <vbank> [config] [resume]`, and wait for 2000 years over :laughing:
    - e.g. `make_preprocess.cmd ljspeech default`
  - or if you wants to perform step by step, refer to recipes in `Makefile`:
    - `make dirs VBANK=<vbank> WAVPATH=<wavpath>` creates necessary folder hierachy and soft-links
    - `make units VBANK=<vbank>` encodes wavforms to hubert's hidden-units
    - `make mels VBANK=<vbank>` transforms wavforms to log-mel spectrograms 
    - `make train VBANK=<vbank> CONFIG=[config] RESUME=[resume]` trains the acoustic model with paired data (unit, mel)
    - `make train_resume VBANK=<vbank> CONFIG=[config]` resumes training on the saved `model-best.pt`
  - NOTE: preprocessed features are generated in `data\<vbank>\*`, while model checkpoints are saved in `out\<vbank>`
5. you can launch TensorBoard summary by `make stats VBANK=<vbank>`
6. once train finished, run `python infer.py <vbank> <input>` (e.g. `python infer.py ljspeech test`) to generate converted wavfiles for folder `<input>`

If you have neither `make.exe` nor `cmd.exe`, you can directly use the python scripts:

```cmd
# Assure you have created the directory hierachy:
# mkdir data/<vbank> data/<vbank>/units data/<vbank>/mels out
# mklink /J data/<vbank>/wavs path/to/vbank/wavpath

python preprocess.py vbank <--encode|melspec>
python train.py vbank --config CONFIG [--resume RESUME]
python infer.py vbank input [--out_path OUT_PATH]
```


### Project Layout

```
.
├── thesis/                   // 参考用原始论文
├── acoustic/                 // 声学模型代码
├── configs/                  // 训练用超参数配置
│   ├── default.json
│   ├── <config>.json
│   └── ...
├── data/                     // 训练用数据文件
│   ├── <vbank>/
│   │   ├── wavs/             // 指向<wavpath>的目录软连接 (由mklink产生)
│   │   ├── units/            // preprocess产生的HuBERT特征
│   │   └── mels/             // preprocess产生的Mel谱特征
│   └── ...
├── out/                      // 模型权重保存点 + 日志统计
│   ├── <vbank>/
│   │   ├── logs/             // 日志(`*.log`) + TFBoard(`events.out.tfevents.*`)
│   │   ├── model-best.pt     // 最优检查点
│   │   ├── model-<steps>.pt  // 中间检查点
│   └── ...
├── preprocess.py             // 数据预处理代码
├── train.py                  // 训练代码
├── infer.py                  // 合成代码 (Commandline API)
├── demo.ipynb                // 编程API示例 (Programmatic API)
|── ...
├── make_train.cmd            // 自定义语音库预处理脚本 (仅预处理，步骤1~3)
├── make_preprocess.cmd       // 自定义语音库训练脚本 (仅训练，步骤4)
├── Makefile                  // 自定义语音库任务脚本 (分步骤)
|── ...
├── test/                     // demo源数据集
├── gen/                      // demo生成数据集 (demo源数据集在demo声库上产生的转换结果)
├── index.html                // demo列表页面
├── make_index.py             // demo页面生成脚本 (产生index.html)
└── make_infer_test.cmd       // demo生成数据集生成脚本 (产生gen/)
```

ℹ️ These developed scripts and tools are targeted mainly for **Windows** platform, if you work on Linux or Mac, you possibly need to modify on your own :(

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
