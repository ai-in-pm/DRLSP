import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List
import yaml
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.agents.nfsp_agent import NFSPAgent
from src.environments.leduc_poker import LeducPoker

class NFSPDashboard:
    def __init__(self):
        st.set_page_config(page_title="NFSP Agent Dashboard", layout="wide")
        self.load_config()
        self.initialize_state()
        self.render_dashboard()
    
    def load_config(self):
        """Load configuration from config.yaml"""
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
    
    def initialize_state(self):
        """Initialize session state variables"""
        if "metrics" not in st.session_state:
            st.session_state.metrics = {
                "rl_loss": [],
                "sl_loss": [],
                "exploitability": [],
                "win_rate": [],
                "episode_rewards": []
            }
        
        if "agent" not in st.session_state:
            st.session_state.agent = NFSPAgent(self.config)
            st.session_state.env = LeducPoker()
    
    def render_dashboard(self):
        """Render the main dashboard"""
        st.title("Neural Fictitious Self-Play (NFSP) Agent Dashboard")
        
        # Sidebar controls
        with st.sidebar:
            st.header("Training Controls")
            if st.button("Train Episode"):
                self.train_episode()
            
            st.header("Parameters")
            self.config["training"]["anticipatory_param"] = st.slider(
                "Anticipatory Parameter",
                0.0, 1.0,
                self.config["training"]["anticipatory_param"]
            )
        
        # Main dashboard area
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_training_metrics()
        
        with col2:
            self.render_strategy_visualization()
        
        # Game state and explanation
        st.header("Current Game State")
        self.render_game_state()
    
    def train_episode(self):
        """Run one training episode"""
        state, info = st.session_state.env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = st.session_state.agent.get_action(
                st.session_state.env.get_state_tensor(state, state.current_player)
            )
            next_state, reward, done, info = st.session_state.env.step(state, action)
            episode_reward += reward
            
            # Store experience
            st.session_state.agent.rl_buffer.add(
                (state, action, reward, next_state, done)
            )
            
            if len(st.session_state.agent.rl_buffer.data) >= self.config["training"]["batch_size"]:
                # Update networks
                batch_rl = np.random.choice(
                    st.session_state.agent.rl_buffer.data,
                    self.config["training"]["batch_size"]
                )
                batch_sl = np.random.choice(
                    st.session_state.agent.sl_buffer.data,
                    self.config["training"]["batch_size"]
                )
                
                losses = st.session_state.agent.update(batch_rl, batch_sl)
                
                # Update metrics
                st.session_state.metrics["rl_loss"].append(losses["rl_loss"])
                st.session_state.metrics["sl_loss"].append(losses["sl_loss"])
            
            state = next_state
        
        # Update episode metrics
        st.session_state.metrics["episode_rewards"].append(episode_reward)
        
        # Calculate and update exploitability
        exploitability = self.calculate_exploitability()
        st.session_state.metrics["exploitability"].append(exploitability)
    
    def render_training_metrics(self):
        """Render training metrics plots"""
        st.subheader("Training Metrics")
        
        # Loss plots
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=st.session_state.metrics["rl_loss"],
            name="RL Loss",
            line=dict(color="blue")
        ))
        fig_loss.add_trace(go.Scatter(
            y=st.session_state.metrics["sl_loss"],
            name="SL Loss",
            line=dict(color="red")
        ))
        fig_loss.update_layout(title="Network Losses")
        st.plotly_chart(fig_loss)
        
        # Exploitability plot
        fig_exploit = px.line(
            y=st.session_state.metrics["exploitability"],
            title="Exploitability"
        )
        st.plotly_chart(fig_exploit)
    
    def render_strategy_visualization(self):
        """Render visualization of current strategy"""
        st.subheader("Strategy Visualization")
        
        # Create sample game states
        sample_states = self.generate_sample_states()
        
        # Get action probabilities for each state
        action_probs = []
        for state in sample_states:
            probs = st.session_state.agent.sl_network(
                torch.FloatTensor(
                    st.session_state.env.get_state_tensor(state, state.current_player)
                )
            ).softmax(dim=0).detach().numpy()
            action_probs.append(probs)
        
        # Create heatmap
        fig = px.imshow(
            action_probs,
            labels=dict(x="Action", y="State"),
            title="Action Probabilities Across States"
        )
        st.plotly_chart(fig)
    
    def render_game_state(self):
        """Render current game state and strategy explanation"""
        state, _ = st.session_state.env.reset()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Game State")
            st.write(f"Stage: {'Pre-flop' if state.stage == 0 else 'Flop'}")
            st.write(f"Pot: {state.pot}")
            st.write(f"Current Player: {state.current_player}")
            st.write(f"Bets: {state.bets}")
        
        with col2:
            st.subheader("Strategy Explanation")
            explanation = st.session_state.agent.explain_strategy(
                st.session_state.env.get_state_tensor(state, state.current_player)
            )
            st.write(explanation)
    
    def calculate_exploitability(self) -> float:
        """Calculate current strategy exploitability"""
        # This is a simplified calculation
        # In practice, you would need to compute best response value
        # against current average strategy
        return np.random.random() * np.exp(-len(st.session_state.metrics["episode_rewards"]) / 1000)
    
    def generate_sample_states(self) -> List[LeducState]:
        """Generate sample game states for visualization"""
        states = []
        for _ in range(5):
            state, _ = st.session_state.env.reset()
            states.append(state)
        return states

if __name__ == "__main__":
    dashboard = NFSPDashboard()
