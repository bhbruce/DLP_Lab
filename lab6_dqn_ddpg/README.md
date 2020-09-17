# Lab6 RL_DQN_DDPG

#### In this lab, implement DQN, DDQN, and DDPG. Using DQN or DDQN to learn how to play LunarLander-V2, and using DDPG to learn how to play LunarLanderContinuous-V2. The environment for simulation is OpenAI GYM simulator. 

## Usage
### (1) DQN / DDQN

#### Train model with 12000 episodes and test.  
```
python dqn.py [--ddqn] [--render] [--test_only]
```
* --test_only: only testing no training
* --render: display current status of env
* --ddqn: if add this flag, use DDQN algorithm.

### (2) DDPG

```
python ddpg.py [--render] [--test_only]
```
## Result

#### Test 10 episodes
|Algorithm| DQN| DDQN| DDPG |
|-------- | -------- | -------- | -------- |
| Avg. Reward | 256     | 247     | 254     |

