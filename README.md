# Transfer-Learning-in-Reinforcement-Learning
Transfer Learning in Reinforcement Learning project

## Project description
Transfer learning approaches in reinforcement learning aim to assist agents in learning their target domains by leveraging the knowledge learned from other agents that have been trained on similar source domains. For example, recent research focus within this space has been placed on knowledge transfer between tasks that have different transition dynamics and reward functions; however, little focus has been placed on knowledge transfer between tasks that have different action spaces. 

In this paper, we approach the task of transfer learning between domains that differ in action spaces. We present a reward shaping method based on source embedding similarity that is applicable to domains with both discrete and continuous action spaces. The efficacy of our approach is evaluated on transfer to restricted action spaces in the Acrobot-v1 and Pendulum-v0 domains (Brockman et al. 2016). 
## Our presentations

* Presentation 1 [here](https://docs.google.com/presentation/d/1BcU8_edTa50EC6Cxv-vd-XJShPOH5gRgiisUbYtAJXI/edit#slide=id.p)
* Google Doc Folder [here](https://drive.google.com/drive/folders/17gtCWIyYdYkFkTXkSYCy4eERVDjVqPDb)

## Our Google Colab

https://colab.research.google.com/drive/1cQCV9Ko-prpB8sH6FlB4oj781On-ut_w?usp=sharing


## Install GYM

Using pip:

```
pip install gym
```

Or Building from Source

```angular2html
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

* We may need to install some dependencies, based on the hints when we run the `test.py`.
* Don't add `gym` source code to our repository

<!-- # Install Mujoco (optional)

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
https://www.chenshiyu.top/blog/2019/06/19/Tutorial-Installation-and-Configuration-of-MuJoCo-Gym-Baselines/ -->

<!-- # Install Stable-Baseline3 (optional)

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
``` -->


## How to run?

### Run with python IDE

1. Open `main.py`  or `main_multiple_run.py`
2. Modify `env_name` and `algorithm` that you want to run
3. Modify parameters in `transfer_execute` function if needed
4. Log will be printed out to the terminal and the plotting result will be shown on the new windows.

### Run with Google Colab

Follow our sample in file `Reward_Shaping_TL.ipynb` to run your own colab.

## Implemented Algorithms in Stable-Baseline3 

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| A2C   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:                |
| DDPG  | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| DQN   | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| HER   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| PPO   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:                |
| SAC   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| TD3   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| QR-DQN<sup>[1](#f1)</sup>  | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :x:                               |
| TQC<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :x:                               |
| Maskable PPO<sup>[1](#f1)</sup>   | :x: | :x: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  |

<b id="f1">1</b>: Implemented in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) GitHub repository.

Actions `gym.spaces`:
 * `Box`: A N-dimensional box that containes every point in the action space.
 * `Discrete`: A list of possible actions, where each timestep only one of the actions can be used.
 * `MultiDiscrete`: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * `MultiBinary`: A list of possible actions, where each timestep any of the actions can be used in any combination.



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



## Related papers
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

## Contributors
1. Nathan Beck <nathan.beck@utdallas.edu>
2. Abhiramon Rajasekharan <abhiramon.rajasekharan@utdallas.edu>
3. Trung Hieu Tran <trunghieu.tran@utdallas.edu>
