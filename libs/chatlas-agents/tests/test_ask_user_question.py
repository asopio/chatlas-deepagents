"""Tests for the AskUserQuestion tool."""

import pytest
from chatlas_agents.tools.ask_user_question import (
    AskUserQuestionTool,
    Question,
    Choice,
    Answer,
    AskUserQuestionResult,
    create_ask_user_question_tool,
)


class TestAskUserQuestionModels:
    """Test Pydantic models for validation."""
    
    def test_choice_model(self):
        """Test Choice model creation."""
        choice = Choice(label="Option A", description="First option")
        assert choice.label == "Option A"
        assert choice.description == "First option"
    
    def test_choice_without_description(self):
        """Test Choice with only label."""
        choice = Choice(label="Simple")
        assert choice.label == "Simple"
        assert choice.description == ""
    
    def test_question_model(self):
        """Test Question model creation."""
        question = Question(
            question="What is your preference?",
            header="Preference",
            options=[
                Choice(label="A"),
                Choice(label="B"),
            ]
        )
        assert question.question == "What is your preference?"
        assert question.header == "Preference"
        assert len(question.options) == 2
        assert not question.multi_select
    
    def test_question_min_options_validation(self):
        """Test that questions require at least 2 options."""
        with pytest.raises(ValueError):
            Question(
                question="Test?",
                options=[Choice(label="Only one")]
            )
    
    def test_question_max_options_validation(self):
        """Test that questions accept maximum 4 options."""
        with pytest.raises(ValueError):
            Question(
                question="Test?",
                options=[
                    Choice(label="1"),
                    Choice(label="2"),
                    Choice(label="3"),
                    Choice(label="4"),
                    Choice(label="5"),  # Too many
                ]
            )
    
    def test_question_header_max_length(self):
        """Test that header is limited to 20 characters."""
        with pytest.raises(ValueError):
            Question(
                question="Test?",
                header="This is way too long for a header",
                options=[Choice(label="A"), Choice(label="B")]
            )
    
    def test_answer_model(self):
        """Test Answer model creation."""
        answer = Answer(
            question="Test question?",
            answer="My answer",
            is_other=True
        )
        assert answer.question == "Test question?"
        assert answer.answer == "My answer"
        assert answer.is_other
    
    def test_result_model(self):
        """Test AskUserQuestionResult model."""
        result = AskUserQuestionResult(
            answers=[
                Answer(question="Q1?", answer="A1"),
                Answer(question="Q2?", answer="A2", is_other=True),
            ],
            cancelled=False
        )
        assert len(result.answers) == 2
        assert not result.cancelled


class TestAskUserQuestionTool:
    """Test the AskUserQuestion tool."""
    
    @pytest.mark.asyncio
    async def test_tool_without_callback(self):
        """Test that tool fails gracefully without callback."""
        tool = AskUserQuestionTool()
        questions = [
            Question(
                question="Test?",
                options=[Choice(label="A"), Choice(label="B")]
            )
        ]
        
        result = await tool._arun(questions)
        assert result["cancelled"] is True
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_tool_with_callback(self):
        """Test tool with a mock callback."""
        async def mock_callback(questions):
            return AskUserQuestionResult(
                answers=[
                    Answer(
                        question=questions[0].question,
                        answer="Option A",
                        is_other=False
                    )
                ],
                cancelled=False
            )
        
        tool = create_ask_user_question_tool(mock_callback)
        questions = [
            Question(
                question="What's your preference?",
                header="Pref",
                options=[Choice(label="Option A"), Choice(label="Option B")]
            )
        ]
        
        result = await tool._arun(questions)
        assert not result["cancelled"]
        assert len(result["answers"]) == 1
        assert result["answers"][0]["answer"] == "Option A"
    
    @pytest.mark.asyncio
    async def test_tool_with_multiple_questions(self):
        """Test tool with multiple questions."""
        async def mock_callback(questions):
            return AskUserQuestionResult(
                answers=[
                    Answer(question=q.question, answer=q.options[0].label)
                    for q in questions
                ],
                cancelled=False
            )
        
        tool = create_ask_user_question_tool(mock_callback)
        questions = [
            Question(question="Q1?", options=[Choice(label="A1"), Choice(label="B1")]),
            Question(question="Q2?", options=[Choice(label="A2"), Choice(label="B2")]),
        ]
        
        result = await tool._arun(questions)
        assert not result["cancelled"]
        assert len(result["answers"]) == 2
    
    @pytest.mark.asyncio
    async def test_tool_cancelled_by_user(self):
        """Test tool when user cancels."""
        async def mock_callback(questions):
            return AskUserQuestionResult(answers=[], cancelled=True)
        
        tool = create_ask_user_question_tool(mock_callback)
        questions = [
            Question(question="Q?", options=[Choice(label="A"), Choice(label="B")])
        ]
        
        result = await tool._arun(questions)
        assert result["cancelled"] is True
        assert len(result["answers"]) == 0
    
    @pytest.mark.asyncio
    async def test_tool_with_other_option(self):
        """Test tool when user selects 'Other' and provides custom answer."""
        async def mock_callback(questions):
            return AskUserQuestionResult(
                answers=[
                    Answer(
                        question=questions[0].question,
                        answer="Custom answer",
                        is_other=True
                    )
                ],
                cancelled=False
            )
        
        tool = create_ask_user_question_tool(mock_callback)
        questions = [
            Question(question="Q?", options=[Choice(label="A"), Choice(label="B")])
        ]
        
        result = await tool._arun(questions)
        assert not result["cancelled"]
        assert result["answers"][0]["is_other"] is True
        assert result["answers"][0]["answer"] == "Custom answer"
    
    @pytest.mark.asyncio
    async def test_tool_callback_exception(self):
        """Test tool handles callback exceptions gracefully."""
        async def failing_callback(questions):
            raise RuntimeError("Simulated error")
        
        tool = create_ask_user_question_tool(failing_callback)
        questions = [
            Question(question="Q?", options=[Choice(label="A"), Choice(label="B")])
        ]
        
        result = await tool._arun(questions)
        assert result["cancelled"] is True
        assert "error" in result
    
    def test_sync_run_not_supported(self):
        """Test that synchronous _run raises NotImplementedError."""
        tool = AskUserQuestionTool()
        questions = [
            Question(question="Q?", options=[Choice(label="A"), Choice(label="B")])
        ]
        
        with pytest.raises(NotImplementedError):
            tool._run(questions)
    
    def test_tool_metadata(self):
        """Test tool has correct metadata."""
        tool = AskUserQuestionTool()
        assert tool.name == "ask_user_question"
        assert "ask" in tool.description.lower()
        assert "questions" in tool.description.lower()
        assert not tool.return_direct


class TestCreateAskUserQuestionTool:
    """Test the factory function."""
    
    def test_create_without_callback(self):
        """Test creating tool without callback."""
        tool = create_ask_user_question_tool()
        assert isinstance(tool, AskUserQuestionTool)
        assert tool.user_input_callback is None
    
    def test_create_with_callback(self):
        """Test creating tool with callback."""
        async def my_callback(questions):
            return AskUserQuestionResult(answers=[], cancelled=False)
        
        tool = create_ask_user_question_tool(my_callback)
        assert isinstance(tool, AskUserQuestionTool)
        assert tool.user_input_callback == my_callback
