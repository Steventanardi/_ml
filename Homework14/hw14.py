import gymnasium as gym

def fixed_policy(observation):
    pole_angle = observation[2]
    return 0 if pole_angle < 0 else 1  # 0 = left, 1 = right

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

total_reward = 0

for step in range(1000):  # limit to 1000 steps
    env.render()
    
    action = fixed_policy(observation)
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if terminated or truncated:
        print(f"Episode ended after {step+1} steps. Total Reward: {total_reward}")
        break

env.close()
