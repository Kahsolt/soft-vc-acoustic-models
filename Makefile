PYTHON_BIN=python

DATA_PATH=data\$(VBANK)
OUT_PATH=out\$(VBANK)
CONFIG_PATH=configs\$(VBANK).json

.PHONY: all dirs units mels train _chk_param_vbank _chk_wavpath_exist _chk_cfgfile_exist

all:
	@echo "Usage:"
	@echo "   make dirs  VBANK=<vbank> WAVPATH=<wavpath>"
	@echo "   make units VBANK=<vbank>"
	@echo "   make mels  VBANK=<vbank>"
	@echo "   make train VBANK=<vbank>"

dirs: _chk_param_vbank _chk_wavpath_exist
	mkdir $(DATA_PATH)
	ln -s $(WAVPATH) $(DATA_PATH)\wavs
	mkdir $(DATA_PATH)\units
	mkdir $(DATA_PATH)\mels
	mkdir $(OUT_PATH)

units: _chk_param_vbank
	$(PYTHON_BIN) preprocess.py \
	  --encode \
	  $(DATA_PATH)\wavs \
	  $(DATA_PATH)\units

mels: _chk_param_vbank
	$(PYTHON_BIN) preprocess.py \
	  --melspec \
	  $(DATA_PATH)\wavs \
	  $(DATA_PATH)\mels

train: _chk_param_vbank _chk_cfgfile_exist
	$(PYTHON_BIN) train.py \
	  $(DATA_PATH) \
	  $(OUT_PATH) \
	  --config $(CONFIG_PATH)


_chk_param_vbank:
ifndef VBANK
	@echo "[Error] must provide your VBANK=<voice_bank_name>"
	@exit -1
endif

_chk_wavpath_exist:
ifndef WAVPATH
	@echo "[Error] must provide your WAVPATH=<path/to/wavfiles>"
	@exit -1
endif
ifeq ($(wildcard $(WAVPATH)),)
	@echo "[Error] wavfile folder $(WAVPATH) not exist!"
	@exit -1
endif

_chk_cfgfile_exist:
ifeq ($(wildcard $(CONFIG_PATH)),)
	@echo "[Error] missing config file $(CONFIG_PATH) for vbank $(VBANK)"
	@exit -1
endif
