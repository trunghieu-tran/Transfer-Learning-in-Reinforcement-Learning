from stable_baselines3.common.monitor import Monitor
from tl.utils.model_evaluation import *
from tl.utils.plot_utils import *
from tl.utils.model_generator import *

def transfer_execute(source_env,
                     target_env,
                     algo='DDPG',
                     policy_name='MlpPolicy',
                     step_number=10000,
                     step_number_small=10000,
                     callback_check_freq=200,
                     evaluation_step=100,
                     log_dir_w_TL="/tmp/gym/w_tl/",
                     log_dir_wo_TL="/tmp/gym/wo_tl/",
                     log_dir_w_TL_rs="/tmp/gym/w_tl_rs/",
                     log_dir_w_full_TL_rs="/tmp/gym/w_full_tl_rs/",
                     run_evaluation=False
                     ):
    print(">>Executing with algorithm " + algo + "...")

    os.makedirs(log_dir_w_TL, exist_ok=True)
    os.makedirs(log_dir_wo_TL, exist_ok=True)
    os.makedirs(log_dir_w_TL_rs, exist_ok=True)
    os.makedirs(log_dir_w_full_TL_rs, exist_ok=True)

    ### ==================
    ### Train source model
    ### ==================

    source_model = get_model(policy_name, source_env, verbose=2, algo=algo)

    source_model.learn(total_timesteps=step_number)
    source_model.save("./source_model_trained")

    if run_evaluation:
        print(">>[Source] Evaluate trained agent:")
        evaluate(source_model, evaluation_step)

    ### =======================================================
    ### Train target model by using our reward shaping approach
    ### =======================================================
    
    # Reload the source model, which will effectively reset the replay buffer
    source_model = source_model.load("./source_model_trained")
    source_model.set_env(source_env)
    
    # Get monitor of the target environment.
    target_env_monitor_rs = Monitor(target_env, log_dir_w_TL_rs)
    
    # Not needed: callback_w_TL_rs = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_w_TL_rs)
    
    # Retrieve a reward shaping model on the target environment
    target_reward_reshaping_model = get_reward_shaping_model(policy_name=policy_name, env=target_env_monitor_rs,
                                                            src_model=source_model, verbose=2, algo=algo,
                                                            num_sampling_episodes=10)
    
    # Learn this reward shaping model on the target environment
    target_reward_reshaping_model.learn(total_timesteps=step_number_small)
    if run_evaluation:
        print(">>[Target] Evaluate trained agent with TL and Reward Shaping:")
        evaluate(target_reward_reshaping_model, evaluation_step)

    ### =====================================================================
    ### Train target model by applying source model directly in target domain
    ### =====================================================================

    target_model = source_model.load("./source_model_trained")

    # As the environment is not serializable, we need to set a new instance of the environment
    target_env_monitor_with_TL = Monitor(target_env, log_dir_w_TL)
    target_model.set_env(target_env_monitor_with_TL)

    # Not needed: callback_w_TL = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_w_TL)
    
    # Continue training on the target model
    target_model.learn(step_number_small)
    if run_evaluation:
        print(">>[Target] Evaluate trained agent using source model:")
        evaluate(target_model, evaluation_step)

    ### ==================================================
    ### Train target model from scratch (without transfer)
    ### ==================================================

    # Get monitor of the target environment as before
    target_env_monitor = Monitor(target_env, log_dir_wo_TL)
    
    # Not needed: callback = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_wo_TL)
    
    # Get a blank model and learn again
    target_model_wo_TL = get_model(policy_name, target_env_monitor, verbose=2, algo=algo)
    target_model_wo_TL.learn(total_timesteps=step_number_small)
    if run_evaluation:
        print(">>[Target] Evaluate trained agent without TL:")
        evaluate(target_model_wo_TL, evaluation_step)


    """
    # Get monitor of the target environment as before. Also, create a copy of the source model that is used in reshaping.
    reshaping_source_model = copy.deepcopy(source_model)

    target_env_monitor_rs2 = Monitor(target_env, log_dir_w_full_TL_rs)
    
    # Not needed: callback_w_full_TL_rs = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_w_full_TL_rs)
    
    
    target_reward_reshaping_model2 = get_reward_shaping_model(policy_name=policy_name, env=target_env_monitor_rs2,
                                                            src_model=loaded_src_model2, verbose=2, algo=algo,
                                                            num_sampling_episodes=10)

    target_reward_reshaping_model2.learn(total_timesteps=step_number_small, callback=callback_w_full_TL_rs)

    if run_evaluation:
        print(">>[Target] Evaluate trained agent with full TL and Reward Shaping:")
        evaluate(target_reward_reshaping_model2, evaluation_step)
    """