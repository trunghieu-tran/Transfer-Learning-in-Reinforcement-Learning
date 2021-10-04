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
```

Or Building from Source

```angular2html
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

* We may need to install some dependencies, based on the hints when we run the `test.py`.
* Don't add `gym` source code to our repository

## Refercences

1. OpenAI Gym [repo](https://github.com/openai/gym)
2. OpenAI Gym [website](https://gym.openai.com/)
3. Example code of TL in DL [repo](https://github.com/anksng/Transfer-learning-in-Deep-Reinforcement-learning)
4. Retro Contest - a transfer learning contest that measures a reinforcement learning algorithm’s ability to generalize from previous experience (hosted by OpenAI) [link](https://openai.com/blog/retro-contest/)


### Related papers
1. Transfer Learning for Related Reinforcement Learning Tasks via Image-to-Image Translation ([paper](https://arxiv.org/pdf/1806.07377.pdf)), ([repo](https://github.com/ShaniGam/RL-GAN))
2. Deep Transfer Reinforcement Learning for Text Summarization ([paper](https://arxiv.org/abs/1810.06667)),([repo](https://github.com/yaserkl/TransferRL)) 
3. Using Transfer Learning Between Games to Improve Deep Reinforcement Learning Performance and Stability ([paper](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Paper.pdf)), ([poster](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Poster.pdf))
4. Multi-Source Policy Aggregation for Transfer Reinforcement Learning between Diverse Environmental Dynamics (IJCAI 2020) ([repo](https://github.com/Mohammadamin-Barekatain/multipolar))
5. Using Transfer Learning Between Games to Improve Deep Reinforcement Learning Performance and Stability ([paper](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Paper.pdf)), ([poster](https://web.stanford.edu/class/cs234/past_projects/2017/2017_Asawa_Elamri_Pan_Transfer_Learning_Poster.pdf))

## Our presentations

* Presentation 1 (Check file in our github repo)
