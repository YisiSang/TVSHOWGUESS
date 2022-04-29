# TVShowGuess #

We release the dataset and the codes for our NAACL 2022 paper: *[TVShowGuess: Character Comprehension in Stories as Speaker Guessing](https://arxiv.org/pdf/2204.07721.pdf)*.


## Environment Setup ##
Our implementation is based on the models and codes supported by [Huggingface](https://github.com/huggingface/transformers).
```bash
conda create -n tvshowguess python=3.7
conda activate tvshowguess
pip install transformers==4.12
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

## Data Preparation ##
1. Download the dataset and move the zip file into the [dataset](dataset/) folder.
2. Extract the data by the following commands
```bash
cd dataset/
unzip split_scenes.zip -d split_scenes
``` 


## Training ##
```bash
python run_longformer.py                \
    --data_dir=dataset/                 \
    --gradient_accumulation_steps=6     \
    --batch_size=2                      \
    --num_epochs=4                      \
    --train
```
The new model will be saved in the [trained_model](trained_model/) folder and named "pytorch_model.pt".


## Evaluation ##
```bash
python run_longformer.py                \
    --data_dir=dataset/                 \
    --batch_size=2                      \
    --test                              \
    --splits dev test                   \
    --from_pretrained=trained_models/SR_Longformer.l2000_w256.MTL.pt
```
The command should reproduce our results in Table 5. One should be able to see the following in the terminal:
```console
========== Evaluation on dev ==========
[*] SHOW NAME - FRIENDS:
    Person-Level Dev Acc: 0.7701
[*] SHOW NAME - The_Big_Bang_Theory:
    Person-Level Dev Acc: 0.6387
[*] SHOW NAME - Frasier:
    Person-Level Dev Acc: 0.9032
[*] SHOW NAME - Gilmore_Girls:
    Person-Level Dev Acc: 0.8217
[*] SHOW NAME - The_Office:
    Person-Level Dev Acc: 0.7181
[*] Average Person-Level Dev Acc: 0.7695
========== END of Evaluation ==========
```


## Citation ##
If you find this repo useful, please consider citing our paper:
```bibtex
@article{sang2022tvshowguess,
  title={TVShowGuess: Character Comprehension in Stories as Speaker Guessing},
  author={Sang, Yisi and Mou, Xiangyang and Yu, Mo and Yao, Shunyu and Li, Jing and Stanton, Jeffrey},
  journal={arXiv preprint arXiv:2204.07721},
  year={2022}
}
```