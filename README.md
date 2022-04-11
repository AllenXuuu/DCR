# DCR

This repo contains the official implementation of paper


> Learning to Anticipate Future with Dynamic Context Removal    
> [Xinyu Xu](https://xuxinyu.website), [Yong-Lu Li](https://dirtyharrylyl.github.io/), [Cewu Lu](https://mvig.sjtu.edu.cn).
>
> In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 2022.
> 
> [[arxiv](https://arxiv.org/abs/2204.02587)] [[code](https://github.com/AllenXuuu/DCR)] [[model](https://drive.google.com/drive/folders/1bXFs1_9HBPi74LpsYfxx753Vkc6BbEHa?usp=sharing)]
****

## Data Prepare

We reorganize the annotation files of four datasets[1-4] in ```data``` folder.   
You need to download pre-extracted feature into ```data/feature``` folder.    
TSN feature can be downloaded from [Link](https://github.com/fpv-iplab/rulstm) [5].   
irCSN-152 feature can be downloaded from [Link](https://github.com/facebookresearch/AVT) [6].    
We provide a stronger TSM backbone, feature can be be downloaded from [Link](https://drive.google.com/drive/folders/1spwT8r7Fcm1fJJFju_L7NdyyHKckODNo?usp=sharing).


## Packages

We conduct experiments in the following environment
```
python == 3.9

torch == 1.9
torchvision == 0.10.0
apex == 0.1.0
tensorboardX
yacs
pyyaml
numpy
prefetch_generator
```


## Evaluation

We release pre-trained models at [here](https://drive.google.com/drive/folders/1bXFs1_9HBPi74LpsYfxx753Vkc6BbEHa?usp=sharing).  
To test the performance of our model, for example using RGB-TSM backbone on EPIC-KITCHENS-100[1], you can run the following command.

```
python eval.py --cfg configs/EK100RGBTSM/eval.yaml --resume ./weights/EK100RGBTSM.pt
```

Here ```./weights/EK100RGBTSM.pt``` is the path to the pre-trained model you downloaded.

More results can be found in [Model Zoo](./docs/model_zoo.md).

## Training 

Taking the same setting as an example, to reproduce the training process, you can run

```
python train_order.py --cfg configs/EK100RGBTSM/order.yaml --name order

python train.py --cfg configs/EK100RGBTSM/train.yaml --name train --resume exp/EK100RGBTSM/order/epoch_50.pt
```

The first line runs our frame order pre-training stage. The model will be stored in ```exp/EK100RGBTSM/order/epoch_50.pt```. The second line reloads the pre-trained model and runs the anticipation training stage.

Only to do the anticipation training from scratch is also possible by

```
python train.py --cfg configs/EK100RGBTSM/train.yaml --name train 
```


## Citation

If you find our paper or code helpful, please cite our paper.

```
@inproceedings{xu2022learning,
  title={Learning to Anticipate Future with Dynamic Context Removal },
  author={Xu, Xinyu and Li, Yong-Lu and Lu, Cewu},
  booktitle={CVPR},
  year={2022}
}
```


## Reference

**[1]** Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision (IJCV), 2021.

**[2]** Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. Scaling egocentric vision: The epic-kitchens dataset. In Proceedings of the European Conference on Computer Vision (ECCV), pages 720–736, 2018.

**[3]** Yin Li, Miao Liu, and James M. Rehg. In the eye of beholder: Joint learning of gaze and actions in first person video. In Proceedings of the European Conference on Computer Vision (ECCV), September 2018.


**[4]** Sebastian Stein and Stephen J McKenna. Combining embedded accelerometers with computer vision for recognizing food preparation activities. In Proceedings of the 2013 ACM international joint conference on Pervasive and ubiquitous computing, pages 729–738, 2013.

**[5]** Antonino Furnari and Giovanni Farinella. Rolling-unrolling lstms for action anticipation from first-person video. IEEE transactions on pattern analysis and machine intelligence, 2020.


**[6]** Rohit Girdhar and Kristen Grauman. Anticipative Video Transformer. In ICCV, 2021.

