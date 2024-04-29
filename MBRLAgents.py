#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-based Reinforcement Learning policies
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
By Thomas Moerland
"""
import numpy as np
from queue import PriorityQueue
from MBRLEnvironment import WindyGridworld

class DynaAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_sa_s = np.zeros((n_states, n_actions, n_states))  
        self.R_sum_sa_s = np.zeros((n_states, n_actions, n_states))  
        self.visits_sa = np.zeros((n_states, n_actions))  

    def select_action(self, s, epsilon):
        if np.random.rand() < epsilon:
            a = np.random.randint(0, self.n_actions)
        else:
            a = np.argmax(self.Q_sa[s])
        return a
        
    def update(self, s, a, r, done, s_next, n_planning_updates):
        target = r + self.gamma * np.max(self.Q_sa[s_next]) * (not done)
        self.Q_sa[s][a] += self.learning_rate * (target - self.Q_sa[s][a])
        self.n_sa_s[s][a][s_next] += 1
        self.R_sum_sa_s[s][a][s_next] += r
        self.visits_sa[s][a] += 1
        
        # Planning
        for _ in range(n_planning_updates):
            s_sim = np.random.choice(self.n_states, p=self.visits_sa[:, :].sum(axis=1) / self.visits_sa.sum())
            a_sim = np.random.choice(self.n_actions, p=self.visits_sa[s_sim] / self.visits_sa[s_sim].sum())
            sampled_transitions = self.n_sa_s[s_sim][a_sim]
            total_transitions = sampled_transitions.sum()
            if total_transitions > 0:
                probabilities = sampled_transitions / total_transitions
                s_next_sim = np.random.choice(self.n_states, p=probabilities)
                r_sim = self.R_sum_sa_s[s_sim][a_sim][s_next_sim] / sampled_transitions[s_next_sim]
                target_sim = r_sim + self.gamma * np.max(self.Q_sa[s_next_sim])
                self.Q_sa[s_sim][a_sim] += self.learning_rate * (target_sim - self.Q_sa[s_sim][a_sim])

    def evaluate(self, eval_env, n_eval_episodes=30, max_episode_length=100):
        returns = []
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s])
                s_next, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                s = s_next
            returns.append(R_ep)
        return np.mean(returns)

class PrioritizedSweepingAgent:
    def __init__(self, n_states, n_actions, learning_rate, gamma, priority_cutoff=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.priority_cutoff = priority_cutoff  
        self.queue = PriorityQueue()

        self.Q_sa = np.zeros((n_states, n_actions))
        self.n_sa_s = np.zeros((n_states, n_actions, n_states))
        self.R_sum_sa_s = np.zeros((n_states, n_actions, n_states))
        self.visits_sa = np.zeros((n_states, n_actions))
        self.model = {s: {a: [] for a in range(n_actions)} for s in range(n_states)}

    def select_action(self, s, epsilon):
        # TO DO: Change this to e-greedy action selection

        if np.random.rand() < epsilon:
            action = np.random.randint(self.n_actions)
        else:
            action = np.argmax(self.Q_sa[s])
        return action
        
    def update(self,s,a,r,done,s_next,n_planning_updates):
        self.n_sa_s[s][a][s_next] += 1
        self.R_sum_sa_s[s][a][s_next] += r
        self.visits_sa[s][a] += 1
        self.model[s][a].append((s_next, r)) 

        # Update Q-value
        max_q_next = np.max(self.Q_sa[s_next]) if not done else 0
        self.Q_sa[s][a] += self.learning_rate * (r + self.gamma * max_q_next - self.Q_sa[s][a])

        # Compute priority
        priority = abs(r + self.gamma * max_q_next - self.Q_sa[s][a])
        if priority > self.priority_cutoff:  
            self.queue.put((-priority, (s, a))) 

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        for i in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for t in range(max_episode_length):
                a = np.argmax(self.Q_sa[s]) 
                s_prime, r, done = eval_env.step(a)
                R_ep += r
                if done:
                    break
                else:
                    s = s_prime
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return        

def test():

    n_timesteps = 10001
    gamma = 1.0

    # Algorithm parameters
    policy = 'ps' # or 'ps' 
    epsilon = 0.1
    learning_rate = 0.2
    n_planning_updates = 3

    # Plotting parameters
    plot = True
    plot_optimal_policy = True
    step_pause = 0.0001
    
    # Initialize environment and policy
    env = WindyGridworld()
    if policy == 'dyna':
        pi = DynaAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize Dyna policy
    elif policy == 'ps':    
        pi = PrioritizedSweepingAgent(env.n_states,env.n_actions,learning_rate,gamma) # Initialize PS policy
    else:
        raise KeyError('Policy {} not implemented'.format(policy))
    
    # Prepare for running
    s = env.reset()  
    continuous_mode = False
    
    for t in range(n_timesteps):            
        # Select action, transition, update policy
        a = pi.select_action(s,epsilon)
        s_next,r,done = env.step(a)
        pi.update(s=s,a=a,r=r,done=done,s_next=s_next,n_planning_updates=n_planning_updates)
        
        # Render environment
        if plot:
            env.render(Q_sa=pi.Q_sa,plot_optimal_policy=plot_optimal_policy,
                       step_pause=step_pause)
            
        # Ask user for manual or continuous execution
        if not continuous_mode:
            key_input = input("Press 'Enter' to execute next step, press 'c' to run full algorithm")
            continuous_mode = True if key_input == 'c' else False

        # Reset environment when terminated
        if done:
            s = env.reset()
        else:
            s = s_next
            
    
if __name__ == '__main__':
    test()
