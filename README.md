# BSSE-SE
This is the official implementation of our paper *"Boosting Self-Supervised Embeddings for Speech Enhancement"*

## Requirements
- pytorch 1.10.2
- torchaudio 0.10.2
- pesq 0.0.3
- pystoi 0.3.3
- numpy 1.20.3
- tensorboardx 2.2
- tqdm 4.60.0
- scikit-learn 0.24.1
- pandas 1.2.4

You can use pip to install Python depedencies.

```
pip install -r requirements.txt
```

## Data preparation

#### Voice Bank--Demand Dataset
The Voice Bank--Demand Dataset is not provided by this repository. Please download the dataset and build your own PyTorch dataloader from [here](https://datashare.is.ed.ac.uk/handle/10283/1942?show=full).
For each `.wav` file, you need to first convert it into 16kHz format by any audio converter (e.g., [sox](http://sox.sourceforge.net/)).
```
sox <48K.wav> -r 16000 -c 1 -b 16 <16k.wav>
```

#### Pretrained enhancement model weight
Please download the model weights from [here](https://drive.google.com/drive/folders/1cwDoGdF44ExQt__B6Z44g3opUdH-hJXE?usp=sharing), and make a folder named `save_model` then put the weight file under the folder. 

#### Result on Voice Bank--Demand
Experiment Date | PESQ | CSIG | CBAK | COVL
-|-|-|-|-
2022-04-30 | 3.20 | 4.52 | 3.58 | 3.88

## Usage
Run the following command to train the speech enhancement model:
```
python main.py \
    --data_folder <root/dir/of/dataset> 
    --model BLSTM 
    --feature <log1p/ssl/cross> 
    --size <base/large> 
    --target IRM 
    --finetune_SSL <PF/EF/None> 
    --weighted_sum
```

add `--mode test` in the command line and the rest remain the same to evaluate the speech enhancement model:
```python main.py --mode test ```


## Citation
Please cite the following paper if you find the codes useful in your research.

```
@article{hung2022boosting,
  title={Boosting Self-Supervised Embeddings for Speech Enhancement},
  author={Hung, Kuo-Hsuan and Fu, Szu-wei and Tseng, Huan-Hsin and Chiang, Hsin-Tien and Tsao, Yu and Lin, Chii-Wann},
  journal={arXiv preprint arXiv:2204.03339},
  year={2022}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
