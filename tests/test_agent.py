"""
Tests for the Advanced Agentic RAG with Chain-of-Thought reasoning.

Run with: python -m pytest tests/test_agent.py -v
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock langchain modules before importing agent
sys.modules['langchain'] = MagicMock()
sys.modules['langchain.agents'] = MagicMock()
sys.modules['langchain.memory'] = MagicMock()
sys.modules['langchain_ollama'] = MagicMock()
sys.modules['dotenv'] = MagicMock()
sys.modules['tools'] = MagicMock()
sys.modules['httpx'] = MagicMock()

from agent import (
    ThoughtStep,
    ReasoningTrace,
    ChainOfThoughtReasoner,
    CriticAgent,
    SpecializedAgent,
    ManagerAgent,
    AdvancedAgenticRAG,
    AgenticRAG,
    check_ollama_connection,
    get_available_model,
    OLLAMA_BASE_URL,
    DEFAULT_MODEL
)


# ============== Fixtures ==============

@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    llm = Mock()
    llm.invoke = Mock(return_value="Mock response")
    llm.ainvoke = AsyncMock(return_value="Mock async response")
    return llm


@pytest.fixture
def sample_reasoning_response():
    """Sample LLM response for CoT reasoning."""
    return """
STEP 1: [Understand the question]
Thought: The user is asking about Python programming basics.
Confidence: 0.9

STEP 2: [Break down the problem]
Thought: I need to identify key Python concepts to explain.
Confidence: 0.85

STEP 3: [Identify information needed]
Thought: I should retrieve documentation about Python fundamentals.
Confidence: 0.8

STEP 4: [Analyze and reason]
Thought: Python is a high-level programming language with simple syntax.
Confidence: 0.9

STEP 5: [Synthesize answer]
Thought: I will provide a clear explanation of Python basics.
Confidence: 0.85

FINAL_ANSWER: Python is a versatile programming language known for its readability.
TOTAL_CONFIDENCE: 0.86
"""


@pytest.fixture
def sample_evaluation_response():
    """Sample LLM response for critic evaluation."""
    return """
ACCURACY: 8
RELEVANCE: 9
COMPLETENESS: 7
REASONING_QUALITY: 8
CLARITY: 9
OVERALL_SCORE: 8.2
VERDICT: ACCEPT
FEEDBACK: Good response with clear explanation.
"""


# ============== ThoughtStep Tests ==============

class TestThoughtStep:
    def test_thought_step_creation(self):
        """Test ThoughtStep dataclass creation."""
        step = ThoughtStep(
            step_number=1,
            thought="Test thought",
            confidence=0.8
        )
        assert step.step_number == 1
        assert step.thought == "Test thought"
        assert step.confidence == 0.8
        assert step.action is None
        assert step.observation is None

    def test_thought_step_with_action(self):
        """Test ThoughtStep with action and observation."""
        step = ThoughtStep(
            step_number=2,
            thought="Analyzing data",
            action="search_documents",
            observation="Found 3 relevant documents",
            confidence=0.75
        )
        assert step.action == "search_documents"
        assert step.observation == "Found 3 relevant documents"


# ============== ReasoningTrace Tests ==============

class TestReasoningTrace:
    def test_reasoning_trace_creation(self):
        """Test ReasoningTrace dataclass creation."""
        steps = [
            ThoughtStep(1, "First thought", confidence=0.8),
            ThoughtStep(2, "Second thought", confidence=0.9)
        ]
        trace = ReasoningTrace(
            query="What is Python?",
            steps=steps,
            final_answer="Python is a programming language.",
            total_confidence=0.85
        )
        assert trace.query == "What is Python?"
        assert len(trace.steps) == 2
        assert trace.total_confidence == 0.85


# ============== ChainOfThoughtReasoner Tests ==============

class TestChainOfThoughtReasoner:
    def test_reasoner_initialization(self, mock_llm):
        """Test ChainOfThoughtReasoner initialization."""
        reasoner = ChainOfThoughtReasoner(mock_llm)
        assert reasoner.llm == mock_llm

    @pytest.mark.asyncio
    async def test_reason_basic(self, mock_llm, sample_reasoning_response):
        """Test basic reasoning capability."""
        mock_llm.ainvoke = AsyncMock(return_value=sample_reasoning_response)
        reasoner = ChainOfThoughtReasoner(mock_llm)

        trace = await reasoner.reason("What is Python?")

        assert trace.query == "What is Python?"
        assert len(trace.steps) > 0
        assert trace.total_confidence > 0
        assert "Python" in trace.final_answer


    def test_parse_reasoning_trace_malformed(self, mock_llm):
        """Test parsing handles malformed responses gracefully."""
        reasoner = ChainOfThoughtReasoner(mock_llm)

        malformed_response = "This is just a plain text response without structure."
        trace = reasoner._parse_reasoning_trace("Test query", malformed_response)

        # Should still create a trace with fallback values
        assert trace.query == "Test query"
        assert len(trace.steps) >= 1
        assert trace.total_confidence == 0.5  # Default confidence


# ============== CriticAgent Tests ==============

class TestCriticAgent:
    def test_critic_initialization(self, mock_llm):
        """Test CriticAgent initialization."""
        critic = CriticAgent(mock_llm)
        assert critic.llm == mock_llm

    def test_evaluate_basic(self, mock_llm, sample_evaluation_response):
        """Test basic evaluation."""
        mock_llm.invoke = Mock(return_value=sample_evaluation_response)
        critic = CriticAgent(mock_llm)

        result = critic.evaluate(
            message="What is Python?",
            response="Python is a programming language."
        )

        assert result["accuracy"] == 8
        assert result["relevance"] == 9
        assert result["verdict"] == "ACCEPT"

    def test_evaluate_with_reasoning_trace(self, mock_llm, sample_evaluation_response):
        """Test evaluation with reasoning trace context."""
        mock_llm.invoke = Mock(return_value=sample_evaluation_response)
        critic = CriticAgent(mock_llm)

        trace = ReasoningTrace(
            query="What is Python?",
            steps=[ThoughtStep(1, "Analyzing question", confidence=0.8)],
            final_answer="Python is a language",
            total_confidence=0.8
        )

        result = critic.evaluate(
            message="What is Python?",
            response="Python is a programming language.",
            reasoning_trace=trace
        )

        # Verify reasoning was included in prompt
        call_args = mock_llm.invoke.call_args[0][0]
        assert "Reasoning Steps Used" in call_args
        assert "confidence: 0.8" in call_args

    def test_parse_evaluation_scores(self, mock_llm):
        """Test parsing of evaluation scores."""
        critic = CriticAgent(mock_llm)

        response = """
ACCURACY: 7
RELEVANCE: 8
COMPLETENESS: 6
REASONING_QUALITY: 7
CLARITY: 8
OVERALL_SCORE: 7.2
VERDICT: IMPROVE
FEEDBACK: Need more detail on Python syntax.
"""
        result = critic._parse_evaluation(response)

        assert result["accuracy"] == 7
        assert result["overall_score"] == 7.2
        assert result["verdict"] == "IMPROVE"
        assert "Python syntax" in result["feedback"]


# ============== AdvancedAgenticRAG Tests ==============

class TestAdvancedAgenticRAG:
    @patch('agent.get_available_model', return_value='mistral')
    @patch('agent.OllamaLLM')
    @patch('agent.RAGTool')
    @patch('agent.SQLTool')
    @patch('agent.CalculatorTool')
    @patch('agent.APITool')
    def test_initialization(self, mock_api, mock_calc, mock_sql, mock_rag, mock_ollama, mock_model):
        """Test AdvancedAgenticRAG initialization."""
        mock_ollama.return_value = Mock()

        agent = AdvancedAgenticRAG(
            model_name="mistral",
            temperature=0.7,
            max_refinements=3,
            min_confidence_threshold=0.6
        )

        assert agent.max_refinements == 3
        assert agent.min_confidence_threshold == 0.6
        assert agent.reasoner is not None
        assert agent.reasoning_history == []

    @patch('agent.get_available_model', return_value='mistral')
    @patch('agent.OllamaLLM')
    @patch('agent.RAGTool')
    @patch('agent.SQLTool')
    @patch('agent.CalculatorTool')
    @patch('agent.APITool')
    def test_reasoning_history_tracking(self, mock_api, mock_calc, mock_sql, mock_rag, mock_ollama, mock_model):
        """Test that reasoning history is tracked."""
        mock_ollama.return_value = Mock()

        agent = AdvancedAgenticRAG()

        # Verify empty history at start
        assert len(agent.get_reasoning_history()) == 0

        # Clear history
        agent.clear_reasoning_history()
        assert len(agent.reasoning_history) == 0

    @patch('agent.get_available_model', return_value='mistral')
    @patch('agent.OllamaLLM')
    @patch('agent.RAGTool')
    @patch('agent.SQLTool')
    @patch('agent.CalculatorTool')
    @patch('agent.APITool')
    def test_format_reasoning_for_context(self, mock_api, mock_calc, mock_sql, mock_rag, mock_ollama, mock_model):
        """Test formatting of reasoning trace for context."""
        mock_ollama.return_value = Mock()

        agent = AdvancedAgenticRAG()

        trace = ReasoningTrace(
            query="Test query",
            steps=[
                ThoughtStep(1, "First thought", confidence=0.8),
                ThoughtStep(2, "Second thought", confidence=0.9)
            ],
            final_answer="Test answer",
            total_confidence=0.85
        )

        context = agent._format_reasoning_for_context(trace)

        assert "Step 1: First thought" in context
        assert "Step 2: Second thought" in context
        assert "Confidence: 0.85" in context


# ============== Backward Compatibility Tests ==============

class TestBackwardCompatibility:
    @patch('agent.get_available_model', return_value='mistral')
    @patch('agent.OllamaLLM')
    @patch('agent.RAGTool')
    @patch('agent.SQLTool')
    @patch('agent.CalculatorTool')
    @patch('agent.APITool')
    def test_agentic_rag_is_advanced(self, mock_api, mock_calc, mock_sql, mock_rag, mock_ollama, mock_model):
        """Test AgenticRAG is an alias for AdvancedAgenticRAG."""
        mock_ollama.return_value = Mock()

        agent = AgenticRAG()

        assert isinstance(agent, AdvancedAgenticRAG)
        assert hasattr(agent, 'reasoner')
        assert hasattr(agent, 'reasoning_history')


# ============== Integration Tests ==============

class TestIntegration:
    @pytest.mark.asyncio
    @patch('agent.get_available_model', return_value='mistral')
    @patch('agent.OllamaLLM')
    @patch('agent.RAGTool')
    @patch('agent.SQLTool')
    @patch('agent.CalculatorTool')
    @patch('agent.APITool')
    @patch('agent.initialize_agent')
    async def test_full_chat_flow(self, mock_init_agent, mock_api, mock_calc,
                                   mock_sql, mock_rag, mock_ollama, mock_model):
        """Test complete chat flow with mocked components."""
        # Setup mocks
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(return_value="""
STEP 1: Understanding
Thought: User asks about Python.
Confidence: 0.9

FINAL_ANSWER: Python is a programming language.
TOTAL_CONFIDENCE: 0.9
""")
        mock_llm.invoke = Mock(return_value="""
ACCURACY: 9
RELEVANCE: 9
COMPLETENESS: 8
REASONING_QUALITY: 9
CLARITY: 9
OVERALL_SCORE: 8.8
VERDICT: ACCEPT
FEEDBACK: Good response.
""")
        mock_ollama.return_value = mock_llm

        # Mock the agent's run method
        mock_agent_instance = Mock()
        mock_agent_instance.arun = AsyncMock(return_value="Python is great!")
        mock_init_agent.return_value = mock_agent_instance

        agent = AdvancedAgenticRAG()

        # Verify agent was created with reasoner
        assert agent.reasoner is not None


# ============== Ollama Integration Tests ==============

class TestOllamaIntegration:
    """Tests for Ollama LLM integration."""

    def test_default_model_is_mistral(self):
        """Test that default model is mistral."""
        assert DEFAULT_MODEL == "mistral"

    def test_ollama_base_url_default(self):
        """Test default Ollama base URL."""
        assert "localhost" in OLLAMA_BASE_URL
        assert "11434" in OLLAMA_BASE_URL

    @patch('agent.httpx')
    def test_check_ollama_connection_success(self, mock_httpx):
        """Test successful Ollama connection check."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "mistral:latest"},
                {"name": "llama2:latest"}
            ]
        }
        mock_httpx.get.return_value = mock_response

        result = check_ollama_connection()
        assert result["connected"] == True
        assert "mistral:latest" in result["models"]

    @patch('agent.httpx')
    def test_check_ollama_connection_failure(self, mock_httpx):
        """Test Ollama connection failure."""
        mock_httpx.get.side_effect = Exception("Connection refused")

        result = check_ollama_connection()
        assert result["connected"] == False
        assert "error" in result

    @patch('agent.check_ollama_connection')
    def test_get_available_model_preferred(self, mock_check):
        """Test getting preferred model when available."""
        mock_check.return_value = {
            "connected": True,
            "models": ["mistral:latest", "llama2:latest"]
        }

        result = get_available_model("mistral")
        assert result == "mistral"

    @patch('agent.check_ollama_connection')
    def test_get_available_model_fallback(self, mock_check):
        """Test fallback to available model."""
        mock_check.return_value = {
            "connected": True,
            "models": ["codellama:latest"]
        }

        result = get_available_model("mistral")
        assert result == "codellama"

    @patch('agent.check_ollama_connection')
    def test_get_available_model_not_connected(self, mock_check):
        """Test error when Ollama not connected."""
        mock_check.return_value = {"connected": False, "models": []}

        with pytest.raises(ConnectionError):
            get_available_model("mistral")

    @patch('agent.check_ollama_connection')
    def test_get_available_model_no_models(self, mock_check):
        """Test error when no models available."""
        mock_check.return_value = {"connected": True, "models": []}

        with pytest.raises(ValueError):
            get_available_model("mistral")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
