import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

@dataclass
class ReplayBuffer:
    capacity: int
    data: List = None
    
    def __post_init__(self):
        self.data = []
        self.count = 0
    
    def add(self, experience):
        if len(self.data) < self.capacity:
            self.data.append(experience)
        else:
            idx = np.random.randint(self.count + 1)
            if idx < self.capacity:
                self.data[idx] = experience
        self.count += 1

class QNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = size
            
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class NFSPAgent:
    def __init__(self, config: Dict):
        self.config = config
        self.rl_buffer = ReplayBuffer(config["training"]["rl_buffer_size"])
        self.sl_buffer = ReplayBuffer(config["training"]["sl_buffer_size"])
        
        # Initialize LLM for strategy explanation
        self.llm = ChatOpenAI(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"]
        )
        
        # Initialize networks
        input_dim = 14  # state dimension
        hidden_dim = config["model"]["hidden_dim"]
        output_dim = 5  # action space size
        
        self.rl_network = QNetwork(input_dim, [hidden_dim], output_dim)
        self.sl_network = PolicyNetwork(input_dim, [hidden_dim], output_dim)
        
        # Optimizers
        self.rl_optimizer = torch.optim.Adam(
            self.rl_network.parameters(),
            lr=config["training"]["rl_learning_rate"]
        )
        
        self.sl_optimizer = torch.optim.Adam(
            self.sl_network.parameters(),
            lr=config["training"]["sl_learning_rate"]
        )
        
        # Load models if they exist
        self._load_models()
    
    def _get_state_size(self) -> int:
        # This should be implemented based on the specific game
        return 100  # placeholder
    
    def _get_action_size(self) -> int:
        # This should be implemented based on the specific game
        return 10  # placeholder
    
    async def get_action(self, state, is_training: bool = True) -> Tuple[int, float]:
        """Get action using anticipatory dynamics"""
        if is_training and np.random.random() < 0.5:
            # Use best response (RL) strategy
            with torch.no_grad():
                q_values = self.rl_network(torch.FloatTensor(state))
                action = torch.argmax(q_values).item()
        else:
            # Use average (SL) strategy
            with torch.no_grad():
                probs = torch.softmax(self.sl_network(torch.FloatTensor(state)), dim=0)
                action = torch.multinomial(probs, 1).item()
        
        return action
    
    async def update(self, batch_rl, batch_sl):
        """Update both networks"""
        # Update RL network (best response)
        states, actions, rewards, next_states, dones = zip(*batch_rl)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Q-learning update
        current_q = self.rl_network(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_q = self.rl_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * 0.99 * next_q
        
        rl_loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.rl_optimizer.zero_grad()
        rl_loss.backward()
        self.rl_optimizer.step()
        
        # Update SL network (average strategy)
        states, actions = zip(*batch_sl)
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        
        action_probs = self.sl_network(states)
        sl_loss = nn.CrossEntropyLoss()(action_probs, actions)
        
        self.sl_optimizer.zero_grad()
        sl_loss.backward()
        self.sl_optimizer.step()
        
        return {
            "rl_loss": rl_loss.item(),
            "sl_loss": sl_loss.item()
        }
    
    async def explain_strategy(self, state) -> str:
        """Use LLM to explain current strategy"""
        action_probs = torch.softmax(self.sl_network(torch.FloatTensor(state)), dim=0)
        
        prompt = f"""
        Given the current game state and action probabilities:
        Action probabilities: {action_probs.tolist()}
        
        Please explain the strategy being employed by the agent, considering:
        1. Which actions are most likely and why
        2. How this relates to game-theoretic principles
        3. Potential counter-strategies the opponent might employ
        """
        
        messages = [
            SystemMessage(content="You are an expert in game theory and poker strategy analysis."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.agenerate([messages])
        return response.generations[0][0].text

    def _load_models(self):
        # Load models if they exist
        pass
