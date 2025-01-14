import asyncio
from tqdm import tqdm
import numpy as np
import torch
import random
import logging
import yaml
from .agents.nfsp_agent import NFSPAgent
from .environments.leduc_poker import LeducPoker

async def train_episode(agent, env, epsilon=0.1):
    state = env.reset()
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy exploration
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = await agent.get_action(state)
        
        next_state, reward, done, info = env.step(action)
        agent.rl_buffer.add((state, action, reward, next_state, done))
        agent.sl_buffer.add((state, action))
        
        state = next_state
        total_reward += reward
        
        if len(agent.rl_buffer.data) >= agent.config["training"]["batch_size"]:
            batch_rl = random.sample(agent.rl_buffer.data, agent.config["training"]["batch_size"])
            batch_sl = random.sample(agent.sl_buffer.data, agent.config["training"]["batch_size"])
            loss_info = await agent.update(batch_rl, batch_sl)
            
    return total_reward

async def main():
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize environment and agent
    env = LeducPoker()
    agent = NFSPAgent(config)
    
    # Training loop
    num_episodes = config["training"]["num_episodes"]
    epsilon = config["training"]["epsilon"]
    
    logger.info("Starting training...")
    progress_bar = tqdm(range(num_episodes))
    
    for episode in progress_bar:
        total_reward = await train_episode(agent, env, epsilon)
        
        if episode % 100 == 0:
            logger.info(f"Episode {episode}/{num_episodes}, Total Reward: {total_reward}")
            
            # Save model checkpoints
            torch.save(agent.rl_network.state_dict(), f"checkpoints/rl_network_{episode}.pt")
            torch.save(agent.sl_network.state_dict(), f"checkpoints/sl_network_{episode}.pt")
            
            # Get strategy explanation
            state = env.reset()
            explanation = await agent.explain_strategy(state)
            logger.info(f"Current strategy explanation: {explanation}")

if __name__ == "__main__":
    asyncio.run(main())
