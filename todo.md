## TODO

1. change the reward in get_data()
2. We have two kinds of data: 
    - real data from env
    - fake data using PILCO
    
    maybe we should store real data for further use
    
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

9. what if s_action is negative ?

10. Save/restore model(gp/actor) ?

11. how to get states to train actor ? random sample: we should sample more states for those appear commonly.
 
