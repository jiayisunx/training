# install dependency package
1. Base library 

```
pip install numpy h5py matplotlib hickle tqdm cffi python-Levenshtein librosa wget tensorboardX wget -i https://pypi.tuna.tsinghua.edu.cn/simple
```
2. sox

install:
```
conda install -c conda-forge sox
```

3. torchaudio

This need gcc version>=5.0
install:
```
conda install -c pytorch torchaudio
```

4. warp-ctc
install: 
```
git clone https://github.com/SeanNaren/warp-ctc.git && \
	cd warp-ctc && \
	mkdir -p build && cd build && cmake .. && make && \
    cd ../pytorch_binding && python setup.py install
```

# dataset
download dataset: ./download_dataset.sh 
you can also use local dataset by symlink like:
```
mkdir -p pytorch/datasets/coco
ln -s /path_to_deepspeech/libri_train_manifest.csv libri_train_manifest.csv
ln -s /path_to_deepspeech/libri_val_manifest.csv libri_val_manifest.csv

```

# download weight file:
```

```

# run training and inference: cd pytorch/

## training
```
./run_and_time.sh
```

## inference
```
./run_inference_multi_instances.sh
```

