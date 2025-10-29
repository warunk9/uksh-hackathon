# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reviser agent for correcting inaccuracies based on verified findings."""

import os
from . import prompt
from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

from dotenv import load_dotenv

load_dotenv()

ask_vertex_dataflow_retrieval = VertexAiRagRetrieval(
    name='dataflow_rag_corpus',
    description=
    ('Use this tool to retrieve documentation and reference materials for the question about dataflow or apache beam or code generation about dataflow or apache beam from the RAG corpus,'
    ),
    rag_resources=[
        rag.RagResource(
            # please fill in your own rag corpus
            # here is a sample rag corpus for testing purpose
            # e.g. projects/123/locations/us-central1/ragCorpora/456
            rag_corpus=os.environ.get("RAG_CORPUS"))
    ],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
)

ask_vertex_dvt_retrieval = VertexAiRagRetrieval(
    name='dvt_data_rag_corpus',
    description=
    ('Use this tool to retrieve documentation and reference materials for the question about data validation from the RAG corpus '
    ),
    rag_resources=[
        rag.RagResource(
            # please fill in your own rag corpus
            # here is a sample rag corpus for testing purpose
            # e.g. projects/123/locations/us-central1/ragCorpora/456
            rag_corpus=
            "projects/agent-pikachu-dev/locations/us-central1/ragCorpora/2266436512474202112"
        )
    ],
    similarity_top_k=5,
    vector_distance_threshold=0.4,
)

ask_vertex_dataproc_retrieval = VertexAiRagRetrieval(
    name='ddataproc_rag_corpus',
    description=
    ('Use this tool to retrieve documentation and reference materials for the question about dataproc templates from the RAG corpus '
    ),
    rag_resources=[
        rag.RagResource(
            # please fill in your own rag corpus
            # here is a sample rag corpus for testing purpose
            # e.g. projects/123/locations/us-central1/ragCorpora/456
            rag_corpus=
            "projects/agent-pikachu-dev/locations/us-central1/ragCorpora/3054003497310617600"
        )
    ],
    similarity_top_k=5,
    vector_distance_threshold=0.4,
)

pso_tool_code_helper = Agent(model=str(os.getenv("GOOGLE_GENAI_MODEL")),
                             name='pso_data_tools_code_helper',
                             instruction=prompt.PSO_DATA_TOOLS_CODE_HELPER,
                             tools=[
                                 ask_vertex_dataflow_retrieval,
                                 ask_vertex_dataproc_retrieval,
                                 ask_vertex_dvt_retrieval
                             ])
