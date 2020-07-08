# class MDP:
#     def __init__(self, states, actions, transition_model, reward_fn,
#                      discount_factor = 1.0, start_dist = None):
#         # states: list or set of states
#         # actions: list or set of actions
#         # transition_model: function from (state, action) into DDist over next state
#         # reward_fn: function from (state, action) to real-valued reward
#         # discount_factor: real, greater than 0, less than or equal to 1
#         # start_dist: optional instance of DDist, specifying initial state dist
#         #    if it's unspecified, we'll use a uniform over states
#     def terminal(self, s):
#         # Given a state, return True if the state should be considered to
#         # be terminal.  You can think of a terminal state as generating an
#         # infinite sequence of zero reward.
#     def init_state(self):
#         # Return an initial state by drawing from the distribution over start states.
#     def sim_transition(self, s, a):
#         # Simulates a transition from the given state, s and action a, using the
#         # transition model as a probability distribution.  If s is terminal,
#         # use init_state to draw an initial state.  Returns (reward, new_state)

# class TabularQ:
#     def __init__(self, states, actions):
#         self.actions = actions
#         self.states = states
#         self.q = dict([((s, a), 0.0) for s in states for a in actions])
#     def copy(self):
#         q_copy = TabularQ(self.states, self.actions)
#         q_copy.q.update(self.q)
#         return q_copy
#     def set(self, s, a, v):
#         self.q[(s,a)] = v
#     def get(self, s, a):
#         return self.q[(s,a)]

#####

#Q update
# data is a list of (s, a, t) tuples.
# lr is a learning rate (\alphaα above)
# We will have to update self.q[(s,a)] for all of the data.

def update(self, data, lr):
    # Your code here
    for t in data:
      self.set(t[0],t[1],(1-lr)*self.get(t[0],t[1])+lr*t[-1])
    return None

def Q_learn(mdp, q, lr=.1, iters=100, eps = 0.5, interactive_fn=None):
    s0= mdp.init_state()
    for i in range(iters):
        if interactive_fn: interactive_fn(q, i)
        if mdp.terminal(s0):
            a = epsilon_greedy(q,s0)
            r, sp = mdp.sim_transition(s0,a)
            target2 = r
            tup = (s0, a, target2)
            q.update([tup],lr) 
            s0=sp
        else:
            a = epsilon_greedy(q,s0)
            r, sp = mdp.sim_transition(s0,a)
            target = r + mdp.discount_factor*value(q,sp)
            tup = (s0, a, target)
            q.update([tup],lr) 
            s0=sp
     # don't touch this line
    return q

    # evaluate this cell so you can use the definition in your code below

# def sim_episode(mdp, episode_length, policy, draw=False):
#     '''
#     Simulate an episode (sequence of transitions) of at most
#     episode_length, using policy function to select actions.  If we find
#     a terminal state, end the episode.  Return accumulated reward a list
#     of (s, a, r, s') where s' is None for transition from terminal state.
#     Also return an animation if draw=True, or None if draw=False
#     '''
#     episode = []
#     reward = 0
#     s = mdp.init_state()
#     all_states = [s]
#     for i in range(episode_length):
#         a = policy(s)
#         (r, s_prime) = mdp.sim_transition(s, a)
#         reward += r
#         if mdp.terminal(s):
#             episode.append((s, a, r, None))
#             break
#         episode.append((s, a, r, s_prime))
#         if draw: 
#             mdp.draw_state(s)
#         s = s_prime
#         all_states.append(s)
#     animation = animate(all_states, mdp.n, episode_length) if draw else None
#     return reward, episode, animation

def Q_learn_batch(mdp, q, lr=.1, iters=100, eps=0.5,episode_length=10, n_episodes=2,
interactive_fn=None):
    all_experiences = []
    for i in range(iters):
        if interactive_fn: interactive_fn(q, i)
        for i in range(n_episodes):
            sim = sim_episode(mdp, episode_length, lambda s: epsilon_greedy(q,s,eps))
            for t in sim[1]:
                all_experiences.append(t)
        all_q_targets=[]
        for t in all_experiences:
            if t[-1]==None:
                tup = (t[0],t[1],t[2])
                all_q_targets.append(tup)
            else:
                tup = (t[0],t[1],t[2]+mdp.discount_factor*value(q,t[-1]))
                all_q_targets.append(tup)

        q.update(all_q_targets,lr)
    return q

    class NNQ:
        def __init__(self, states, actions, state2vec, num_layers, num_units, lr=1e-2, epochs=1):
            self.actions = actions
            self.states = states
            self.state2vec = state2vec
            self.epochs = epochs
            self.lr = lr
            state_dim = state2vec(states[0]).shape[1] # a row vector
            self.models = {a : make_nn(state_dim, num_layers, num_units)for a in actions}
            # Your code here
            self.running_loss = 0.
            self.running_one = 0.
            self.num_running = 0.001
        def predict(self, model, s):
          return model(torch.FloatTensor(self.state2vec(s))).detach().numpy()
        def get(self, s, a):
            return self.predict(self.models[a],s)
        def fit(self, model, X,Y, epochs=None, dbg=None):
          if epochs is None: epochs = self.epochs
          train = torch.utils.data.TensorDataset(torch.FloatTensor(X), torch.FloatTensor(Y))
          train_loader = torch.utils.data.DataLoader(train, batch_size=256,shuffle=True)
          opt = torch.optim.SGD(model.parameters(), lr=self.lr)
          for epoch in range(epochs):
            for (X,Y) in train_loader:
              opt.zero_grad()
              loss = torch.nn.MSELoss()(model(X), Y)
              loss.backward()
              self.running_loss = self.running_loss*(1.-self.num_running) + loss.item()*self.num_running
              self.running_one = self.running_one*(1.-self.num_running) + self.num_running
              opt.step()
          if dbg is True or (dbg is None and np.random.rand()< (0.001*X.shape[0])):
            print('Loss running average: ', self.running_loss/self.running_one)
        def update(self, data, lr, dbg=None):
            for a in self.actions:
              if [s for (s,at,t) in data if a==at]:
                X = np.vstack([self.state2vec(s) for (s, at, t) in data if a==at])
                Y = np.vstack([t for (s, at, t) in data if a==at])
                self.fit(self.models[a],X,Y, epochs=self.epochs, dbg=dbg)