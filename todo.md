## TODO

1. change the reward in get_data()
2. We have two kinds of data: 
    - real data from env
    - fake data using PILCO
    
    maybe we should store real data for further use -- in PILCO
    
3. test actor in each/several iteration, then plot ---Done
4. implement other model-free/model-based RL:
    - PG/AC/PPO
    - PILCO/...

5. other env: mujoco/Atari ? :
    - add more reward class
    - update actor network architecture
    
6. action variance too big ? add regularization on loss

7. discount factor gamma to add in get_data()

8. maybe add a critic network

9. what if s_action is negative ?  --Done

10. Save/restore model(gp/actor) ? --Done

11. how to get states to train actor ? random sample: we should sample more states for those appear commonly.

12. change take_action() for other discrete action space

13. how does mgpr optimize: set_XY() ? should it contain old data ?

14. reward calculation ? not accuracy, find another way to calculate ?

15. discrete case: transition model does not receives the same (training compare with prediction)

16. np.nan when calculate M_dx, S_dx ?

17. replay buffer. 

18. for each step in real env, we can use this step to optimize our actor. and we can test actor performance every k steps

19. implementation of PPO, TRPO ...

20. more pilco/gp models to provide expected reward ? (when the actor is good enough, and when we use actor to collect data
for gaussian process, these transitions are not useful, maybe we should use several GPs to give a more stable transition function)

21. memory leak prob for continuous adding ops in graph. maybe rewrite get_data() in pilco.py or figure out a better
way to collect fake data.

22. the result of gppg is not good, possible reasons:
    - the policy in gp is deterministe because it outputs always the same action distribution and which will result in a 
    same next obs distribution and average reward 
    - it is not stable to optimize actor only with fake data, maybe we should add some real data in optimization step
    - maybe the variance of action is too big, we will add a l2 regularisation for variance of action.
    - we should not sample an action from action distribution because we pass action distribution to calculate culmulative reward
    so, it is actually the action distribution that result in the culmulative reward instead of a particular action. So, we 
    should find another way to sample action or find another way to optimize the actor(maybe similar in DDPG or Deep PILCO)

23. since what we do here should use action distribution for optimization, so we need actually find a better way for optim.
for example, we should apply an algorithm only for action distribution. -- maybe some changes on REINFORCE.
