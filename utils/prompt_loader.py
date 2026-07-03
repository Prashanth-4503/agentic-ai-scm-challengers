import yaml
from pathlib import Path
from utils.config import BASE_DIR

PROMPTS_DIR = BASE_DIR / "prompts"

_agents_config = {}
_tasks_config = {}


def load_prompts():
    """Load agents and tasks prompt configurations from YAML files."""
    global _agents_config, _tasks_config
    agents_path = PROMPTS_DIR / "agents.yaml"
    tasks_path = PROMPTS_DIR / "tasks.yaml"

    if agents_path.exists():
        with open(agents_path, "r", encoding="utf-8") as f:
            _agents_config = yaml.safe_load(f) or {}

    if tasks_path.exists():
        with open(tasks_path, "r", encoding="utf-8") as f:
            _tasks_config = yaml.safe_load(f) or {}


def get_agent_prompt(agent_key: str) -> dict:
    """Retrieve agent prompt dictionary containing role, goal, backstory."""
    if not _agents_config:
        load_prompts()
    return _agents_config.get(agent_key, {})


def get_task_prompt(task_key: str) -> dict:
    """Retrieve task prompt dictionary containing description, expected_output."""
    if not _tasks_config:
        load_prompts()
    return _tasks_config.get(task_key, {})
