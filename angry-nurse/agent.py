

"""LLM Auditor for verifying & refining LLM-generated answers using the web."""
import os
import logging
import warnings
from google.adk.agents import Agent  # type: ignore
from dotenv import load_dotenv
from .root_agent_prompt import return_instructions_root

warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic.*")

logger = logging.getLogger(__name__)
load_dotenv()

root_agent = Agent(
    model=str(os.getenv("GOOGLE_GENAI_MODEL")),
    name='root_agent',
    instruction=return_instructions_root(),
    description=('Angry Nurse Agent'),
    sub_agents=[
    ],
)
