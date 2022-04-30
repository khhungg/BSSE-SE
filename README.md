# BSSE-SE
This is the official implementation of our paper *"Boosting Self-Supervised Embeddings for Speech Enhancement"*

## Requirements
- pytorch 1.6
- torchcontrib 0.0.2
- torchaudio 0.6.0
- pesq 0.0.1
- colorama 0.4.3
- fairseq 0.9.0
- geomloss 0.2.3

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

#### Enhancement model pretrained weight
Please download the model weights from [here](https://drive.google.com/drive/folders/1cwDoGdF44ExQt__B6Z44g3opUdH-hJXE?usp=sharing), and make a folder named `save_model` then put the weight file under the folder. 

#### Result on Voice Bank--Demand
Experiment Date | PESQ | CSIG | CBAK | COVL
-|-|-|-|-
2022-04-30 | 3.20 | 4.52 | 3.58 | 3.88


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments
* [Bio-ASP Lab](https://bio-asplab.citi.sinica.edu.tw), CITI, Academia Sinica, Taipei, Taiwan
