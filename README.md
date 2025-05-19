Rocket landing simulator using Proximal Policy Optimization

policy controls engine thrust toggle, as well as engine angle. i also tried adding reaction control thrusters on the top of the rocket in order to better control angular velocity, but this didnt train as well, but i still got something decent.
if you look on the right, you should see a probability distribution over actions that the policy will take. this works in update steps, in which the policy looks at a state vector, including position, velocity, angle, etc, and returns a probability distribution over actions to take. on runtime this distribution is sampled from at every timestep in order to create optimal behaviour. this policy is trained by penalizing trajectories that do not get close to the landing pad, and is given more reward for getting closer to the landing pad, at lower velocity, and more upright angle. the single biggest challenge with this project was designing the reward function in such a way that it allowed for good exploration during training. training was also very slow because i used CPU

the readability could certainly be improved, it is not good. i was inexperienced when i made this.

https://github.com/aerioy/PPO-rocket-land/assets/93295441/367b09ca-0d11-4f69-9141-3ce539d18557

https://github.com/aerioy/PPO-rocket-land/assets/93295441/7c44b6c6-cdd3-4482-a078-6929fc6d1212
