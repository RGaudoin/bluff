"""
Agent factory for creating agents by type name.

Provides a registry of available agent types and factory functions
for creating agents. Supports both BaseAgent (for BluffGame) and
lazy loading of trained models.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Any
import random

from bluff.agents.base import BaseAgent
from bluff.agents.random_agent import RandomAgent
from bluff.agents.heuristic_agent import HeuristicAgent


# Type alias for agent factory functions
AgentFactory = Callable[[str, int, Dict[str, Any]], BaseAgent]


class AgentRegistry:
    """
    Registry of available agent types.

    Agents can be registered with a name and factory function.
    The factory function receives (policy_id, seat, config) and returns a BaseAgent.
    """

    def __init__(self):
        self._factories: Dict[str, AgentFactory] = {}
        self._descriptions: Dict[str, str] = {}
        self._requires_config: Dict[str, List[str]] = {}

    def register(
        self,
        name: str,
        factory: AgentFactory,
        description: str = "",
        requires: Optional[List[str]] = None,
    ) -> None:
        """
        Register an agent type.

        Args:
            name: Display name for the agent type
            factory: Function(policy_id, seat, config) -> BaseAgent
            description: Human-readable description
            requires: List of required config keys (e.g., ["model_path"])
        """
        self._factories[name] = factory
        self._descriptions[name] = description
        self._requires_config[name] = requires or []

    def create(
        self,
        name: str,
        seat: int,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """
        Create an agent of the given type.

        Args:
            name: Registered agent type name
            seat: Seat number for the agent
            config: Optional configuration dict

        Returns:
            BaseAgent instance
        """
        if name not in self._factories:
            raise ValueError(f"Unknown agent type: {name}. Available: {list(self._factories.keys())}")

        config = config or {}
        policy_id = f"{name.lower().replace(' ', '_')}_{seat}"
        return self._factories[name](policy_id, seat, config)

    def list_types(self) -> List[str]:
        """Get list of registered agent type names."""
        return list(self._factories.keys())

    def get_description(self, name: str) -> str:
        """Get description for an agent type."""
        return self._descriptions.get(name, "")

    def get_required_config(self, name: str) -> List[str]:
        """Get required config keys for an agent type."""
        return self._requires_config.get(name, [])

    def is_available(self, name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if an agent type is available with the given config.

        For example, DQN requires a model file to exist and specific game config.
        """
        if name not in self._factories:
            return False

        required = self._requires_config.get(name, [])
        config = config or {}

        for key in required:
            if key not in config:
                return False
            # Special handling for file paths
            if key.endswith("_path"):
                path = Path(config[key])
                if not path.exists():
                    return False

        # Special check for DQN: verify model matches game config
        if "DQN" in name:
            model_path = config.get("model_path")
            if model_path:
                from bluff.rl import DQNPolicy
                metadata = DQNPolicy.load_metadata(str(model_path))
                if metadata:
                    # Check if model config matches game config
                    model_players = metadata.get("num_players")
                    model_dice = metadata.get("dice_per_player")
                    game_players = config.get("num_players", 2)
                    game_dice = config.get("dice_per_player", 3)
                    if model_players != game_players or model_dice != game_dice:
                        return False

        return True


# Global registry instance
_registry = AgentRegistry()


def get_registry() -> AgentRegistry:
    """Get the global agent registry."""
    return _registry


# --- Built-in agent factories ---

def _create_random(policy_id: str, seat: int, config: Dict[str, Any]) -> BaseAgent:
    seed = config.get("seed", random.randint(0, 100000))
    return RandomAgent(policy_id, seed=seed)


def _create_heuristic(policy_id: str, seat: int, config: Dict[str, Any]) -> BaseAgent:
    seed = config.get("seed", random.randint(0, 100000))
    return HeuristicAgent(policy_id, seed=seed)


def _create_dqn(policy_id: str, seat: int, config: Dict[str, Any]) -> BaseAgent:
    """Create a DQN agent from a saved model.

    The model file should contain metadata about training configuration.
    Falls back to config parameters if metadata is not available.
    """
    from bluff.rl import DQNPolicy, get_flat_obs_dim
    from bluff.agents.wrappers import wrap_for_game

    model_path = config.get("model_path")
    if not model_path:
        raise ValueError("DQN agent requires 'model_path' in config")

    # Try to load metadata from checkpoint
    metadata = DQNPolicy.load_metadata(str(model_path))

    if metadata:
        # Use metadata from checkpoint
        num_players = metadata["num_players"]
        dice_per_player = metadata["dice_per_player"]
        num_faces = metadata["num_faces"]
        max_dice = metadata["max_dice"]
        max_tracked_players = metadata["max_tracked_players"]
        obs_dim = metadata["obs_dim"]
        num_actions = metadata["num_actions"]
    else:
        # Fall back to config or defaults (for old checkpoints)
        num_players = config.get("num_players", 2)
        dice_per_player = config.get("dice_per_player", 3)
        num_faces = 6
        max_tracked_players = config.get("max_tracked_players", 32)
        max_dice = num_players * dice_per_player
        obs_dim = get_flat_obs_dim(num_players, num_faces, max_tracked_players)
        num_actions = max_dice * num_faces + 1

    # Create and load policy
    dqn_policy = DQNPolicy(
        obs_dim=obs_dim,
        num_actions=num_actions,
        num_players=num_players,
        num_faces=num_faces,
        max_dice=max_dice,
        max_tracked_players=max_tracked_players,
        device="cpu",  # Use CPU for inference in app
    )
    dqn_policy.load(str(model_path))
    dqn_policy.set_epsilon(0.0)  # Greedy for play
    dqn_policy.eval_mode()

    # Wrap for game use (use actual game params from config)
    actual_num_players = config.get("num_players", num_players)
    actual_dice_per_player = config.get("dice_per_player", dice_per_player)
    actual_max_dice = actual_num_players * actual_dice_per_player

    return wrap_for_game(
        dqn_policy,
        policy_id=policy_id,
        num_faces=num_faces,
        max_dice=actual_max_dice,
        num_players=actual_num_players,
        max_tracked_players=max_tracked_players,
    )


# Register built-in agents
_registry.register(
    "Human",
    lambda pid, seat, cfg: None,  # Human doesn't need an agent
    description="Human player",
)

_registry.register(
    "Random",
    _create_random,
    description="Random valid moves",
)

_registry.register(
    "Heuristic",
    _create_heuristic,
    description="Probability-based decisions",
)

_registry.register(
    "DQN (trained)",
    _create_dqn,
    description="Deep Q-Network (requires trained model)",
    requires=["model_path"],
)


# --- Convenience functions ---

def list_agent_types() -> List[str]:
    """Get list of all registered agent type names."""
    return _registry.list_types()


def create_agent(
    agent_type: str,
    seat: int,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[BaseAgent]:
    """
    Create an agent of the given type.

    Returns None for "Human" type.
    """
    if agent_type == "Human":
        return None
    return _registry.create(agent_type, seat, config)


def is_agent_available(
    agent_type: str,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Check if an agent type is available."""
    return _registry.is_available(agent_type, config)
