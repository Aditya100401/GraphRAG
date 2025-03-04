from langchain.prompts import PromptTemplate
import yaml

# Load the prompt from the YAML file
def load_prompt(file_path='system_prompt.yaml'):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    return data['template']

# Load the template from the YAML file
PROMPT_TEMPLATE = load_prompt()

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=['actor', 'recipient', 'date'],
)