from abc import abstractmethod, ABCMeta

class BaseAgent(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def agent_init(self, agent_info={}):
        'initiate the agent with its parameters'
        

    @abstractmethod
    def agent_start(self, observation):
        'start the agent by taking first action'
        

    @abstractmethod
    def agent_step(self, reward, observation):
        'take actions upto terminal in episodic case or forever'
        

    @abstractmethod
    def agent_end(self, reward):
        'take the final step/update'
        
    
    @abstractmethod
    def agent_cleanup(self):
        'something like current/last state=None'
        pass


    @abstractmethod
    def agent_message(self, message):
        'pass a message and return a response'
        pass