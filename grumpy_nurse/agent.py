

"""LLM Auditor for verifying & refining LLM-generated answers using the web."""
import os
import logging
import warnings
from google.adk.agents import SequentialAgent  
from dotenv import load_dotenv
from .sub_agents.persona_identification import persona_identification_agent
from .sub_agents.scenario_selection import scenario_selection_agent
from .sub_agents.interaction_play import interaction_play_agent
from .sub_agents.feedback import feedback_agent

warnings.filterwarnings("ignore", category=UserWarning, module=".*pydantic.*")

logger = logging.getLogger(__name__)
load_dotenv()

root_agent = SequentialAgent(
    name='Grumpy Nurse Agent',
    description=('Grumpy Nurse Agent'),
    sub_agents= [
        persona_identification_agent,
        scenario_selection_agent,
        interaction_play_agent,
        feedback_agent
    ],
)

