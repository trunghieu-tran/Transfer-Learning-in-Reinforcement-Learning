# Transfer-Learning-in-Reinforcement-Learning
Transfer Learning in Reinforcement Learning project

## Project description
The goal of such a paper/project would be to offer additional insight into the behavioral facets of different transfer RL algorithms by varying the setting and hyperparameters of each algorithm. We are also currently considering the use of OpenAI's Gym package (https://gym.openai.com/) for a possible list of environments to use. 
 
A rough timeline of the project would be as follows:
 
1. Do a literature survey to derive a list of transfer RL approaches to consider in our comparisons.

2. Formulate a list of behavioral facets that we can measure and analyze. These facets could simply be the average expected reward, or they could be other aspects of the setting.

3. Note the experimental settings of these papers to better understand what has been tried.

4. Devise a list of experiments that utilize settings not examined in the previous step. If there are clear hyperparameters in the approaches, we may consider the effect of varying the hyperparameters on our list of behavioral facets.

5. Run these experiments. Luckily, existing code examples (https://github.com/anksng/Transfer-learning-in-Deep-Reinforcement-learning) suggest that these experiments can be written entirely within Jupyter notebooks, which allows us to run them on Google Colab for additional compute.

6. Analyze these results and begin writing the paper, noting the goal and contribution, related work, process, experiments, etc



## Install GYM

Using pip:

```
pip install gym
pip install Box2D
```

Or Building from Source

```angular2html
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

* We may need to install some dependencies, based on the hints when we run the `test.py`.
* Don't add `gym` source code to our repository

# Install Mujoco

First way:
1. Follow this: https://github.com/openai/mujoco-py/#install-mujoco
2. Then install mujoco-py
   1. pip3 install -U 'mujoco-py<2.1,>=2.0'
   2. You may need to install some depencies (based on the HINT on error message when you run command below)

Other way, just run:
``
pip install mujoco_py==2.0.2.8
``

Unfortunately, I can not run Mujoco in Macbook M1. :(
https://github.com/openai/mujoco-py/issues/605

If you guys use linux, you can follow this
https://www.chenshiyu.top/blog/2019/06/19/Tutorial-Installation-and-Configuration-of-MuJoCo-Gym-Baselines/

# Install Stable-Baseline3

Stable Baselines 3 [repo](https://github.com/DLR-RM/stable-baselines3)
```
pip install 'stable-baselines3[extra]'
```

Install atari games:

```
pip install atari-py
```

In order to import ROMS, you need to download Roms.rar from the [Atari 2600 VCS ROM Collection](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html) and extract the .rar file. Once you've done that, run:

`python -m atari_py.import_roms <path to folder>`

This should print out the names of ROMs as it imports them. The ROMs will be copied to your atari_py installation directory.


For example:
 
```
python -m atari_py.import_roms "/Users/harrytran/OneDrive - The University of Texas at Dallas/Fall 2021/CS 7301/project"
```


## Refercences

1. OpenAI Gym [repo](https://github.com/openai/gym)
2. OpenAI Gym [website](https://gym.openai.com/)
3. Stable Baselines 3 [repo](https://github.com/DLR-RM/stable-baselines3)
4. Robotschool [repo](https://github.com/openai/roboschool)
5. Gyem extension [repos](https://github.com/Breakend/gym-extensions) - This python package is an extension to OpenAI Gym for auxiliary tasks (multitask learning, transfer learning, inverse reinforcement learning, etc.)
6. Example code of TL in DL [repo](https://github.com/anksng/Transfer-learning-in-Deep-Reinforcement-learning)
7. Retro Contest - a transfer learning contest that measures a reinforcement learning algorithm’s ability to generalize from previous experience (hosted by OpenAI) [link](https://openai.com/blog/retro-contest/)
8. Rainbow: Combining Improvements in Deep Reinforcement Learning  ([repo](https://github.com/Kaixhin/Rainbow)), ([paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17204/16680))
9. Experience replay ([link](https://paperswithcode.com/method/experience-replay))
10. Solving RL classic control ([link](https://shiva-verma.medium.com/solving-reinforcement-learning-classic-control-problems-openaigym-1b50413265dd))



### Related papers
1. Transfer Learning for Related Reinforcement Learning Tasks via Image-to-Image Translation ([paper](https://arxiv.org/pdf/1806.07377.pdf)), ([repo](https://github.com/ShaniGam/RL-GAN))
2. Deep Transfer Reinforcement Learning for Text Summarization ([paper](https://arxiv.org/abs/1810.06667)),([repo](https://github.com/yaserkl/TransferRL)) 
3. Using Transfer Learning Between Games to Improve Deep Reinforcement Learning Performance and Stability ([paper](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Paper.pdf)), ([poster](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Poster.pdf))
4. Multi-Source Policy Aggregation for Transfer Reinforcement Learning between Diverse Environmental Dynamics (IJCAI 2020) ([paper](https://arxiv.org/pdf/1909.13111.pdf)), ([repo](https://github.com/Mohammadamin-Barekatain/multipolar))
5. Using Transfer Learning Between Games to Improve Deep Reinforcement Learning Performance and Stability ([paper](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Paper.pdf)), ([poster](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Poster.pdf))
6. Deep Reinforcement Learning and Transfer Learning with Flappy Bird ([paper](https://www.cedrick.ai/pdfs/cs221-report.pdf)), ([poster](https://www.cedrick.ai/pdfs/cs221-poster.pdf))
7. Decoupling Dynamics and Reward for Transfer Learning ([paper](https://arxiv.org/pdf/1804.10689.pdf)), ([repo](https://github.com/facebookresearch/ddr))
8. Progressive Neural Networks ([paper](https://arxiv.org/pdf/1606.04671.pdf))
9. Deep Learning for Video Game Playing ([paper](https://arxiv.org/pdf/1708.07902.pdf))
10. Disentangled Skill Embeddings for Reinforcement Learning ([paper](https://arxiv.org/pdf/1906.09223.pdf))
11. Playing Atari with Deep Reinforcement Learning ([paper](https://arxiv.org/pdf/1312.5602.pdf))
12. Dueling Network Architectures for Deep Reinforcement Learning ([paper](http://proceedings.mlr.press/v48/wangf16.pdf))
13. ACTOR-MIMIC DEEP MULTITASK AND TRANSFER REINFORCEMENT LEARNING ([paper](https://arxiv.org/pdf/1511.06342.pdf))
14. DDPG ([link](https://spinningup.openai.com/en/latest/algorithms/ddpg.html))

## Our presentations

* Presentation 1 (Check file in our github repo)
* Google Doc Folder [here](https://drive.google.com/drive/folders/17gtCWIyYdYkFkTXkSYCy4eERVDjVqPDb)

## Google Colab

https://colab.research.google.com/drive/1NpICoFrJNTNOGjx8E7JcEedIZjbhDjnh

Stable Baselines3 - Getting Started
https://colab.research.google.com/drive/1OTIlqgskIlnj48H0CRCy6mb63H-33sKf?usp=sharing
