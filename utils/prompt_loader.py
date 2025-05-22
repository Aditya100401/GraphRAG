import yaml
from pathlib import Path
from langchain_core.messages import SystemMessage

def load_prompt(yaml_path: str = "prompts/system_prompt.yaml") -> str:
    """
    Reads a YAML file with a top-level 'prompt' key and returns its value.
    """
    data = yaml.safe_load(Path(yaml_path).read_text())
    prompt = data.get("template")

    return SystemMessage(content=prompt)
