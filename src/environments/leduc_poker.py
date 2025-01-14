import numpy as np
import gymnasium as gym
from gymnasium import spaces
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

@dataclass
class GameState:
    current_player: int
    player_hands: List[int]
    community_card: Optional[int]
    pot: int
    stage: int  # 0: pre-flop, 1: flop
    last_action: Optional[int]
    last_raise: int

class LeducPoker(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Game parameters
        self.num_players = 2
        self.num_cards = 6  # 2 sets of K, Q, J
        self.small_blind = 1
        self.big_blind = 2
        self.starting_stack = 100
        
        # Action space: fold (0), call (1), raise (2+)
        self.action_space = spaces.Discrete(5)  # fold, call, raise 2x, 3x, 4x
        
        # Observation space: player cards (6), community card (6), pot (1), stage (1)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(14,),
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        """Reset the environment to initial state"""
        # Initialize deck
        self.deck = list(range(self.num_cards))
        np.random.shuffle(self.deck)
        
        # Deal cards to players
        self.state = GameState(
            current_player=0,
            player_hands=[self.deck.pop() for _ in range(self.num_players)],
            community_card=None,
            pot=self.small_blind + self.big_blind,
            stage=0,
            last_action=None,
            last_raise=self.big_blind
        )
        
        return self._get_observation()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        reward = 0
        done = False
        
        # Convert action to game action
        if action == 0:  # fold
            reward = -self.state.pot / 2  # player loses their contribution to pot
            done = True
            new_state = self.state
        else:
            # Handle call or raise
            new_state = self._apply_action(self.state, action)
            
            if self._is_round_over(new_state):
                if new_state.stage == 0:
                    # Deal community card and move to next stage
                    new_state.community_card = self.deck.pop()
                    new_state.stage = 1
                    new_state.current_player = 0
                    new_state.last_action = None
                else:
                    # Showdown
                    reward = self._get_reward(new_state)
                    done = True
        
        self.state = new_state
        return self._get_observation(), reward, done, {}
    
    def _apply_action(self, state: GameState, action: int) -> GameState:
        """Apply action to current state"""
        new_state = GameState(
            current_player=(state.current_player + 1) % self.num_players,
            player_hands=state.player_hands.copy(),
            community_card=state.community_card,
            pot=state.pot,
            stage=state.stage,
            last_action=action,
            last_raise=state.last_raise
        )
        
        if action == 1:  # call
            new_state.pot += state.last_raise
        else:  # raise
            raise_amount = (action - 1) * self.big_blind
            new_state.pot += raise_amount
            new_state.last_raise = raise_amount
        
        return new_state
    
    def _is_round_over(self, state: GameState) -> bool:
        """Check if the current betting round is over"""
        if state.last_action is None:
            return False
        if state.last_action == 0:  # fold
            return True
        return state.last_action == 1 and state.current_player == 0
    
    def _get_reward(self, state: GameState) -> float:
        """Calculate reward at showdown"""
        player_0_rank = self._get_hand_rank(state.player_hands[0], state.community_card)
        player_1_rank = self._get_hand_rank(state.player_hands[1], state.community_card)
        
        if player_0_rank > player_1_rank:
            return state.pot / 2
        elif player_0_rank < player_1_rank:
            return -state.pot / 2
        else:
            return 0
    
    def _get_hand_rank(self, player_card: int, community_card: Optional[int]) -> int:
        """Get the rank of a player's hand"""
        if community_card is None:
            return player_card
        
        # Pair
        if player_card % 3 == community_card % 3:
            return 10 + player_card % 3
        
        # High card
        return max(player_card % 3, community_card % 3)
    
    def _get_observation(self) -> np.ndarray:
        """Convert game state to observation vector"""
        obs = np.zeros(14)
        
        # Encode player hand
        obs[self.state.player_hands[self.state.current_player]] = 1
        
        # Encode community card
        if self.state.community_card is not None:
            obs[6 + self.state.community_card] = 1
        
        # Encode pot and stage
        obs[12] = self.state.pot / (self.starting_stack * 2)
        obs[13] = self.state.stage
        
        return obs
