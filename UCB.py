import Agent
import numpy as np




class ucbAgent(Agent.BaseAgent):
    def __init__(self):
        self.step_size = None
        self.epsilon = None
        # self.discount = agent_info['discount']         # no discount here
        self.num_actions = None
        self.initial_value = None
        self.optimist_init = None
        self.c = None
        self.q_vals = None
        self.last_action = None
        self.ucb_vals = None
        self.num_steps = 0

    def agent_init(self, agent_info = {}):
        self.step_size = agent_info.get('step_size', 0.1)
        self.epsilon = agent_info.get('epsilon', 0.0)
        # self.discount = agent_info['discount']         # no discount here
        self.num_actions = agent_info.get('num_actions', 2)
        self.initial_value = agent_info.get('optimist_value', 0.0)
        self.optimist_init = agent_info.get('is_optimist', False)
        self.actn_counts = np.zeros(self.num_actions)
        self.c = agent_info.get('ucb_const', 0.1)
        
        if self.optimist_init:
            self.q_vals = np.zeros(self.num_actions) * self.initial_value
        else:
            self.q_vals = np.random.random(self.num_actions)
        self.ucb_vals = self.q_vals
    
    def agent_start(self, observation):

        self.last_action = np.random.choice(self.num_actions)
        self.actn_counts[self.last_action] += 1
        self.num_steps += 1

        return self.last_action
    
    def agent_step(self, reward, observation):
        if np.random.random()>self.epsilon:
            current_action = self.argMax(self.ucb_vals)
        else:
            current_action = np.random.choice(self.num_actions)
        old_estimate = self.q_vals[self.last_action]
        self.q_vals[self.last_action] = old_estimate + self.step_size * (reward - old_estimate)
        self.ucb_vals[self.last_action] = self.q_vals[self.last_action] + self.c * np.sqrt(1/(self.actn_counts[self.last_action] * self.step_size))
        
        self.last_action = current_action
        self.actn_counts[self.last_action] += 1
        self.num_steps += 1

        return current_action
    
    def agent_end(self, reward):
        pass

    def agent_cleanup(self):
        pass

    def agent_message(self, message):
        pass
        
    def argMax(self, q_values):
        top_value = -np.inf
        for i in range(len(q_values)):
            if q_values[i]>top_value:
                top_values = [i]
                top_value = q_values[i]
            elif q_values[i]==top_value:
                top_values.append(i)
        return np.random.choice(top_values)        # (uniform) randomly breaking ties