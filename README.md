# GONet: A Semi Supervised Deep Learning Approach For Traversability Estimation
 
**Summary**: Safety is one of the most important topics for the real robot in the real environment. GONet can estimate the traversable probability from the fish eye camera image to avoide the collision. Main contributions of our method is followings,

**I.** Needlessness of the huge annotated untraversable images, which is very hard to collect,  
**II.** Cheaper and stronger estimation than the method using depth information,   
**III.** Realease of our new dataset(http://cvgl.stanford.edu/gonet/dataset/).

Although our method doesn't need the huge annotated untraversable images, the high accuracy can be achieved by a semi supervised deep learning approach based on GAN(Generative Adversarial Network).
Please see the [website](http://cvgl.stanford.edu/gonet/) (http://cvgl.stanford.edu/gonet/) for more technical details. This repository is intended for distribution of the code and its instruction.

#### Paper
**["GONet A Semi Supervised Deep Learning Approach For Traversability Estimation"](http://cvgl.stanford.edu/gonet/)**, in **IROS 2018 [Best Paper Award Finalist on Safety, Security, and Rescue Robotics]**.


[![GONet summary video](misc/gonet_snapplay.png)](https://youtu.be/SmVsGQ2-dlM "Click to watch the video summarizing Gibson environment!")


System Requirement
=================
Ubuntu 16.04

Chainer 4.1.0

Python Pillow 1.1.7

ROS KINETIC(http://wiki.ros.org/kinetic)

Nvidia GPU


Database
=================
Our dataset "GO Stanford" to train GONet is opened at http://cvgl.stanford.edu/gonet/dataset/.

How to use GONet
=================

#### Step1: Choose the method
We have following 4 methods to estimate the traversable probability, depending on your setup. 
The accuracy in the test dataset is GONet-ts(96.90%) > GONet-s(94.90%) > GONet-t(94.45%) > GONet(92.55%).

GONet.py : core GONet using the monocular vision. We can freely choose the frame rate of GONet.

GONet_T.py : GONet-t using the monocular vision with considering the time consistency by LSTM. We recommend to run GONet-t at 3 fps.

GONet_S.py : GONet-s using the stereo vision. We can freely choose the frame rate of GONet-s.

GONet_TS.py : GONet-ts using the stereo vision with considering the time consistency by LSTM. We recommend to run GONet-ts at 3 fps.


#### Step2: Camera Setup
GONet can only accept the fish eye camera image to capture the environment in front of the robot.
And, we highly recommend to use RICOH THETA S, because the training and the evaluation of GONet is done by the collected images by RICOH THETA S.(https://theta360.com/en/about/theta/s.html)

#### Step3: Download


#### Step4: Runing GONet


Citation
=================

If you use GONet's software or database, please cite:
```
@article{hirose2018gonet,
  title={GONet: A Semi-Supervised Deep Learning Approach For Traversability Estimation},
  author={Hirose, Noriaki and Sadeghian, Amir and V{\'a}zquez, Marynel and Goebel, Patrick and Savarese, Silvio},
  booktitle={Intelligent Robots and Systems (IROS), 2018 IEEE/RSJ International Conference on},
  pages={3044--3051},
  year={2018},
  organization={IEEE}
}
```
or
```
@article{hirose2018gonet,
  title={GONet: A Semi-Supervised Deep Learning Approach For Traversability Estimation},
  author={Hirose, Noriaki and Sadeghian, Amir and V{\'a}zquez, Marynel and Goebel, Patrick and Savarese, Silvio},
  journal={arXiv preprint arXiv:1803.03254},
  year={2018}
}
```


