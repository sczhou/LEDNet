<p align="center">
  <img src="assets/LEDNet_LOLBlur_logo.png" height=90>
</p>

## LEDNet: Joint Low-light Enhancement and Deblurring in the Dark (ECCV2022)

[Paper](https://arxiv.org/abs/2202.03373) | [Project Page](https://shangchenzhou.com/projects/LEDNet/) | [Video](https://youtu.be/450dkE-fOMY)

[Shangchen Zhou](https://shangchenzhou.com/), [Chongyi Li](https://li-chongyi.github.io/), [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/) 

S-Lab, Nanyang Technological University

### Updates


- **2022.07.20**:  The code is under preparing, please stay tuned. :blush:
- **2022.07.04**:  This repo is created.




### Dependencies and Installation

- Pytorch >= 1.7.1
- CUDA >= 10.1
- Other required packages in `requirements.txt`
```
# git clone this repository
git clone https://github.com/sczhou/LEDNet
cd LEDNet

# create new anaconda env
conda create -n LEDNet python=3.8 -y
source activate LEDNet

# install python dependencies
pip3 install -r requirements.txt
```

### Citation
If our work is useful for your research, please consider citing:

    @InProceedings{zhou2022lednet,
        author = {Zhou, Shangchen and Li, Chongyi and Loy, Chen Change},
        title = {LEDNet: Joint Low-light Enhancement and Deblurring in the Dark},
        booktitle = {ECCV},
        year = {2022}
    }

### License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.
