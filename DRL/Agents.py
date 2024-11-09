class DDPGAgent:
    """
    Greedily maximize reward signal assuming we know the environment's dynamics
    """

    def __init__(self, model):
        self.model = model

    def policy(self, observation, info, seed=0):
        a = self.model.predict(observation)[0]
#         print(a)
        return a
