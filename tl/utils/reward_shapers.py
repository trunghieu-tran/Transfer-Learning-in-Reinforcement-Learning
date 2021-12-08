from .embeddings import get_ddpg_embeddings, get_dqn_embeddings
from torch import nn

import torch as th

def create_ddpg_reward_shaper(source_model, num_sampling_episodes):
    
    embeddings, q_vals = get_ddpg_embeddings(source_model, num_sampling_episodes)
    reward_shaper = DDPGRewardShaper(source_model, embeddings, q_vals, source_model.gamma)
    return reward_shaper

def create_td3_reward_shaper(source_model, num_sampling_episodes):
    
    return create_ddpg_reward_shaper(source_model, num_sampling_episodes)

def create_dqn_reward_shaper(source_model, num_sampling_episodes):
    
    embeddings, q_vals = get_dqn_embeddings(source_model, num_sampling_episodes)
    reward_shaper = DQNRewardShaper(source_model, embeddings, q_vals, source_model.gamma)
    return reward_shaper

class RewardShaper:

    """
    Base class for computing the auxiliary reward. We extend this class 
    to get the correct embeddings depending on the SB3 model.
    """

    def __init__(self, model, embeddings, associated_q_vals, gamma):

        """
        Creates a RewardShaper instance using the source model, generated 
        embeddings, their associated_q-vals, and the discount factor gamma
        """

        # Store embeddings, their associated q-vals, the learning rate gamma, 
        # and the SB3 model.
        self.embeddings = embeddings.to(model.device)
        self.associated_q_vals = associated_q_vals.to(model.device)
        self.gamma = gamma
        self.model = model

        # Normalize the embeddings. We want to compute cosine similarity, which 
        # is easily done by using the dot product if ||a|| = ||b|| = 1.
        self.embeddings = nn.functional.normalize(self.embeddings, p=2, dim=1)

    def _get_state_action_embedding(self, state, action):
        pass

    def _compute_phi_s_a(self, state, action):

        """
        Computes \Phi(s,a), which is used in the full reward shaping 
        F = \gamma \Phi(s',a') - \Phi(s,a). Here, we compute the 
        state-action embedding
        """

        # Get the embedding for this state-action. Normalize it.
        embedding = self._get_state_action_embedding(state, action)
        embedding = nn.functional.normalize(embedding, p=2, dim=0)

        # Get the dot product of each row of the embedding tensor with this 
        # normalized embedding. As each row vector is also normalized, the 
        # dot product gives exactly the cosine similarity. This can be done 
        # in bulk by doing a matrix product.
        similarity_scores = th.matmul(self.embeddings, embedding)

        # Lastly, we would like an average weighted q-val score. First, we 
        # compute the cosine-sim-weighted q-value sum using the dot product 
        # between our similarity scores and the associated q-vals for each 
        # embedding. Then, we average by the number of embeddings / q-vals
        sum_weighted_q_val_score = th.dot(self.associated_q_vals, similarity_scores)
        avg_weighted_q_val_score = sum_weighted_q_val_score / len(self.associated_q_vals)

        # Return the score.
        return avg_weighted_q_val_score.item()

    def get_auxiliary_reward(self, state, action, next_state, next_action):

        # Get \Phi(s,a) and \Phi(s',a')
        phi_s_a = self._compute_phi_s_a(state, action)
        next_phi_s_a = self._compute_phi_s_a(next_state, next_action)
        
        # Compute auxiliary reward F = \gamma \Phi(s',a') - \Phi(s,a)
        aux_reward = self.gamma * next_phi_s_a - phi_s_a
        return aux_reward


class DDPGRewardShaper(RewardShaper):

    def __init__(self, model, embeddings, associated_q_vals, gamma):

        super(DDPGRewardShaper, self).__init__(model, embeddings, associated_q_vals, gamma)

    def _get_state_action_embedding(self, state, action):

        # Create an alias corresponding to the model's critic network.
        # We will be drawing embeddings from it.
        critic_network = self.model.critic

        # Get the model's policy
        policy = self.model.policy

        # Convert the observation and action to tensor for embedding calculation
        tensor_obs, vectorized_env = policy.obs_to_tensor(state)
        tensor_obs = tensor_obs.to(self.model.device)
        tensor_act = th.reshape(th.tensor(action), (1,-1)).to(self.model.device)

        # Ensure that the model's policy is not training anymore
        policy.set_training_mode(False)

        # Do not keep the gradient computation graph
        with th.no_grad():
      
            # Using the current observation and action to take, get the q-net 
            # embedding and score (of the first network if > 1 exist).
            _, q_embedding = critic_network.q1_forward(tensor_obs, tensor_act, last=True)
                
            # Squeeze the output to get rid of extra dimension
            q_embedding = th.squeeze(q_embedding)
            return q_embedding


class DQNRewardShaper(RewardShaper):

    def __init__(self, model, embeddings, associated_q_vals, gamma):

        super(DQNRewardShaper, self).__init__(model, embeddings, associated_q_vals, gamma)

    def _get_state_action_embedding(self, state, action):

        # Get the model's policy
        policy = self.model.policy

        # Create an alias corresponding to the model's Q-network
        q_network = policy.q_net

        # Ensure that the model's policy is not training anymore
        policy.set_training_mode(False)

        # Convert the observation and action to tensor for embedding calculation
        tensor_obs, vectorized_env = policy.obs_to_tensor(state)
        tensor_obs = tensor_obs.to(self.model.device)

        # Do not keep the gradient computation graph
        with th.no_grad():
      
            # Using the current observation and action to take, get the q-net 
            # embedding and score (of the first network if > 1 exist).
            _, q_embedding = q_network.forward(tensor_obs, last=True)

            # Squeeze the output to get rid of extra dimension
            q_embedding = th.squeeze(q_embedding)

            # Return the embedding
            return q_embedding