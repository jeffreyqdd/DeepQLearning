import os
from lib.agent import Agent
from gym import Env
from tqdm import tqdm

class PrintingUtils:
    def pretty_print(title:str):
        """Pretty print a section header

        ### params
        1. title (str) - section title
        """
        # TODO: add windows support
        os.system('clear')

        # print title
        print( "####" + "#" * max(len(title), 10) )
        print(f"### {title}")
        print( "####" + "#" * max(len(title), 10) )

class EnvWrapper:
    """Clean wrapping for gym environment.
    """
    def run(
        agent:Agent,
        env:Env,
        batch_size:int,
        warmup_steps:int,
        training_steps:int,
        save_every:int=1,
        load_dir:str=None,
        save_dir:str=None,
        render_every=1,
        show_progress=True,
    ):
        """ Runs the entire gym-env training lifecycle. 
        Handles start, warmup, simulation, and stop

        ### Params
        1. agent (Agent) - the agent that trains on the environment
        2. env (Env) - environment
        3. batch_size (int) - batch size for each training epoch
        4. warmup_steps (int) - number of environment steps before training -> to popualte
        5. training_steps (int) - number of training steps
        6. save_every (int) - save every x steps
        7. load_dir (str) - directory for model load
        8. save_dir (str) - directory for model save
        9. render_evetry (int) - render every X resets (-1 --> no render)
        10.show progress (bool) - progress bar
        """
        # Load if possible
        if load_dir is not None:
            agent.load_model(load_dir)
        
        observation = env.reset()

        ### populate memory
        for step in (tqdm(range(warmup_steps), desc="warmup") if show_progress else range(warmup_steps)):
            action_weights              = agent.predict(state=observation)
            action                      = agent.apply_policy(action_weights=action_weights)
            new_obs, reward, done, info = env.step(action)

            if done:
                new_obs = env.reset()

            agent.remember(observation, action, reward, new_obs, done, action_weights)
            observation                 = new_obs

        ### train
        generation = 1
        for step in (tqdm(range(training_steps), desc = "training") if show_progress else range(training_steps)):
            action_weights = agent.predict(state=observation)
            action         = agent.apply_policy(action_weights=action_weights)
            new_obs, reward, done, info = env.step(action)
            
            if render_every > 0 and generation % render_every == 0:
                env.render()

            if done:
                new_obs = env.reset()
                agent.learn(batch_size)
                generation += 1

            agent.remember(observation, action, reward, new_obs, done, action_weights)

            observation = new_obs

            if step % save_every == 0 and save_dir is not None:
                agent.save_model(save_dir)

        env.close()

        # Save if possible
        if save_dir is not None:
            agent.save_model(save_dir)
        