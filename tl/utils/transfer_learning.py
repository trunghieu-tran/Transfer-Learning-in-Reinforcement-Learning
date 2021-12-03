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
                     run_evaluation=False
                     ):
    print(">>Executing with algorithm " + algo + "...")

    os.makedirs(log_dir_w_TL, exist_ok=True)
    os.makedirs(log_dir_wo_TL, exist_ok=True)
    os.makedirs(log_dir_w_TL_rs, exist_ok=True)

    source_model = get_model(policy_name, source_env, verbose=2, algo=algo)
    #
    # print(">>[Source] Evaluate un-trained agent:")
    # evaluate(source_model, evaluation_step)

    source_model.learn(total_timesteps=step_number)
    source_model.save("./source_model_trained")

    if run_evaluation:
        print(">>[Source] Evaluate trained agent:")
        evaluate(source_model, evaluation_step)

    # sample an observation from the environment
    obs = source_model.env.observation_space.sample()

    # Check prediction before saving
    print("pre saved", source_model.predict(obs, deterministic=True))

    # del source_model  # delete trained model to demonstrate loading

    ##### LOAD source model and train with target domain
    target_model = load_model(algo=algo, src="./source_model_trained")
    # Check that the prediction is the same after loading (for the same observation)
    print("loaded", target_model.predict(obs, deterministic=True))

    # as the environment is not serializable, we need to set a new instance of the environment
    target_env_monitor_with_TL = Monitor(target_env, log_dir_w_TL)
    target_model.set_env(target_env_monitor_with_TL)
    callback_w_TL = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_w_TL)
    # and continue training
    target_model.learn(step_number_small, callback=callback_w_TL)
    if run_evaluation:
        print(">>[Target] Evaluate trained agent using source model:")
        evaluate(target_model, evaluation_step)

    #### Train target model without transfer
    target_env_monitor = Monitor(target_env, log_dir_wo_TL)
    callback = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_wo_TL)
    target_model_wo_TL = get_model(policy_name, target_env_monitor, verbose=2, algo=algo)
    target_model_wo_TL.learn(total_timesteps=step_number_small, callback=callback)
    if run_evaluation:
        print(">>[Target] Evaluate trained agent without TL:")
        evaluate(target_model_wo_TL, evaluation_step)

    ##### LOAD source model and train target domain with Reshape
    loaded_src_model = source_model # can not load from file, since env will be empty
    target_env_monitor_rs = Monitor(target_env, log_dir_w_TL_rs)
    callback_w_TL_rs = SaveOnBestTrainingRewardCallback(check_freq=callback_check_freq, log_dir=log_dir_w_TL_rs)
    target_reward_reshaping_model = get_reward_shaping_model(policy_name=policy_name, env=target_env_monitor_rs,
                                                            src_model=loaded_src_model, verbose=2, algo=algo,
                                                            num_sampling_episodes=10)
    target_reward_reshaping_model.learn(total_timesteps=step_number_small, callback=callback_w_TL_rs)
    if run_evaluation:
        print(">>[Target] Evaluate trained agent with TL and Reward Shaping:")
        evaluate(target_reward_reshaping_model, evaluation_step)