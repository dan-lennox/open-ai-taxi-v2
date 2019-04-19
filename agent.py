import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
       
        self.constantAlpha = 0.25
        self.gamma = 1.0
        
    def get_action_probs(self, state, e=0.0):
        """ obtains the action probabilities corresponding to epsilon-greedy policy
        
        Params
        ======
        - state: the current state of the environment
        - e: Threshold for exploitation (0) vs exploration(1).

        Returns
        =======
        - policy_s: The Action to take in the current State.
        """
    
        policy_s = np.ones(self.nA) * e / self.nA
        best_a = np.argmax(self.Q[state])
        policy_s[best_a] = 1 - e + (e / self.nA)
        return policy_s
        
    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        if state not in self.Q:
            return np.random.choice(self.nA)
        return np.random.choice(np.arange(self.nA), p=self.get_action_probs(state, 1/i_episode))

    def step(self, state, next_action, next_reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        
        # SARSA
        prev_estimate = self.Q[state][next_action]
        if (done):
            next_estimate = 0
        else:
            #SARSA
            #next_estimate = self.Q[next_state][next_action]
            # EXPECTED SARSA
            policy_s = self.get_action_probs(state)
            next_estimate = np.dot(self.Q[next_state], policy_s)
            
        # TD Target
        tdTarget = next_reward + self.gamma * next_estimate
        # Set the action value function for the current state and next action.
        self.Q[state][next_action] = prev_estimate + self.constantAlpha * (tdTarget - prev_estimate)