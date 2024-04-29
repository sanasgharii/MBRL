import numpy as np
import matplotlib.pyplot as plt
from MBRLEnvironment import WindyGridworld
from MBRLAgents import DynaAgent, PrioritizedSweepingAgent
from Helper import LearningCurvePlot, smooth

def run_repetitions(agent_class, n_states, n_actions, gamma, learning_rate, epsilon, n_planning_updates, wind_proportion, n_timesteps, eval_interval, n_repetitions, max_episode_length):
    all_evaluations = []  

    for repetition in range(n_repetitions):
        env = WindyGridworld(wind_proportion=wind_proportion)
        agent = agent_class(n_states=env.n_states, n_actions=env.n_actions, learning_rate=learning_rate, gamma=gamma)
        evaluations = []
        timestep = 0
        state = env.reset()

        while timestep < n_timesteps:
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, done, next_state, n_planning_updates)
            timestep += 1

            if timestep % eval_interval == 0 or timestep == n_timesteps:
                eval_score = agent.evaluate(env, n_eval_episodes=30, max_episode_length=max_episode_length)
                evaluations.append((timestep, eval_score))

            state = next_state
            if done:
                state = env.reset()

        all_evaluations.append(evaluations)

    average_evaluations = []
    for t in range(len(all_evaluations[0])):
        timestep, scores = zip(*[(evals[t][0], evals[t][1]) for evals in all_evaluations])
        average_score = np.mean(scores)
        average_evaluations.append((timestep[0], average_score))

    return average_evaluations

def find_best_configuration(results, wind_proportion):
    best_performance = -np.inf
    best_updates = None
    for updates, (timesteps, scores) in results[wind_proportion].items():
        average_performance = np.mean(scores[-10:])
        if average_performance > best_performance:
            best_performance = average_performance
            best_updates = updates
    return best_updates

def plot_best_performers(results, wind_proportions, agents):
    for wind_proportion in wind_proportions:
        plt.figure(figsize=(10, 6))
        for agent_name in agents:
            best_updates = find_best_configuration(results[agent_name], wind_proportion)
            timesteps, scores = results[agent_name][wind_proportion][best_updates]
            plt.plot(timesteps, scores, label=f'Best {agent_name} (Updates: {best_updates})')
        # Also plot the baseline Q-learning which is Dyna with 0 planning updates
        timesteps, scores = results['DynaAgent'][wind_proportion][0]
        plt.plot(timesteps, scores, label='Baseline Q-learning (Updates: 0)', linestyle='--')
        plt.title(f'Best Models Comparison for Wind Proportion {wind_proportion}')
        plt.xlabel('Timesteps')
        plt.ylabel('Average Return')
        plt.legend()
        plt.grid(True)
        plt.show()

def experiment():
    n_timesteps = 10001
    eval_interval = 250
    n_repetitions = 10
    gamma = 1.0
    learning_rate = 0.2
    epsilon = 0.1
    smoothing_window = 10
    n_planning_updates_options = [0, 1, 3, 5]

    wind_proportions = [0.9, 1.0]
    agents = ['DynaAgent', 'PrioritizedSweepingAgent']
    
    results = {agent: {wind: {} for wind in wind_proportions} for agent in agents}

    for agent_name in agents:
        agent_class = globals()[agent_name]
        for wind_proportion in wind_proportions:
            for n_planning_updates in n_planning_updates_options:
                print(f"Running {agent_name} with wind proportion {wind_proportion} and {n_planning_updates} planning updates.")
                evaluations = run_repetitions(agent_class, 70, 4, gamma, learning_rate, epsilon,
                                              n_planning_updates, wind_proportion, n_timesteps,
                                              eval_interval, n_repetitions, 100)
                timesteps, scores = zip(*evaluations)
                results[agent_name][wind_proportion][n_planning_updates] = (timesteps, smooth(scores, smoothing_window))

    # Plot
    for agent_name in agents:
        for wind_proportion in wind_proportions:
            plt.figure(figsize=(10, 6))
            for n_planning_updates, (timesteps, scores) in results[agent_name][wind_proportion].items():
                label = f'{agent_name} (Updates: {n_planning_updates})'
                plt.plot(timesteps, scores, label=label)
            plt.title(f'{agent_name} Learning Curves with Wind Proportion {wind_proportion}')
            plt.xlabel('Timesteps')
            plt.ylabel('Average Return')
            plt.legend()
            plt.grid(True)
            plt.show()

    # best performers comparison plot
    plot_best_performers(results, wind_proportions, agents)

if __name__ == '__main__':
    experiment()