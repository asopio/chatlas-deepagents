"""Interactive user question tool for ChATLAS agents.

This tool allows agents to ask users clarifying questions during execution,
with multi-choice options and free-text fallback.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Callable, Awaitable
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class Choice(BaseModel):
    """A choice option for a question."""
    
    label: str = Field(description="Short label for the choice (1-5 words)")
    description: str = Field(
        default="",
        description="Optional explanation of this choice"
    )


class Question(BaseModel):
    """A question to ask the user."""
    
    question: str = Field(description="The question text")
    header: str = Field(
        default="",
        description="Short header for the question (1-2 words, e.g. 'Auth', 'Database')",
        max_length=20,
    )
    options: list[Choice] = Field(
        description="Available options (2-4, not including 'Other'). An 'Other' option for free text is automatically added.",
        min_length=2,
        max_length=4,
    )
    multi_select: bool = Field(
        default=False,
        description="If true, user can select multiple options"
    )


class AskUserQuestionInput(BaseModel):
    """Input schema for asking user questions."""
    
    questions: list[Question] = Field(
        description="Questions to ask (1-4). Displayed as tabs if multiple.",
        min_length=1,
        max_length=4,
    )


class Answer(BaseModel):
    """User's answer to a question."""
    
    question: str = Field(description="The original question")
    answer: str = Field(description="The user's answer")
    is_other: bool = Field(
        default=False,
        description="True if user typed a custom answer via 'Other'"
    )


class AskUserQuestionResult(BaseModel):
    """Result from asking user questions."""
    
    answers: list[Answer] = Field(description="List of answers")
    cancelled: bool = Field(
        default=False,
        description="True if user cancelled without answering"
    )


# Type for callback function
UserInputCallback = Callable[[list[Question]], Awaitable[AskUserQuestionResult]]


class AskUserQuestionTool(BaseTool):
    """Tool for asking users clarifying questions with multi-choice options.
    
    This tool allows agents to interactively gather user input during execution.
    Each question can have 2-4 options plus an automatic "Other" option for free text.
    
    Examples:
        Ask about preferred approach:
        >>> questions = [Question(
        ...     question="What's the main goal of this refactoring?",
        ...     header="Goal",
        ...     options=[
        ...         Choice(label="Performance", description="Make it run faster"),
        ...         Choice(label="Readability", description="Make it easier to understand"),
        ...         Choice(label="Maintainability", description="Make it easier to modify")
        ...     ]
        ... )]
        
        Multiple questions as tabs:
        >>> questions = [
        ...     Question(question="Which database?", header="DB", 
        ...              options=[Choice(label="PostgreSQL"), Choice(label="MySQL")]),
        ...     Question(question="Which authentication?", header="Auth",
        ...              options=[Choice(label="OAuth2"), Choice(label="JWT")])
        ... ]
    """
    
    name: str = "ask_user_question"
    description: str = (
        "Ask the user one or more questions and wait for their responses. "
        "Each question has 2-4 choices plus an automatic 'Other' option for free text. "
        "Use this to gather preferences, clarify requirements, or get decisions. "
        "Multiple questions are displayed as tabs for better organization."
    )
    args_schema: type[BaseModel] = AskUserQuestionInput
    return_direct: bool = False
    
    # Callback function to handle user input (injected by CLI/UI)
    user_input_callback: Optional[UserInputCallback] = None
    
    def _run(self, questions: list[Question]) -> dict[str, Any]:
        """Synchronous execution - not supported for this tool."""
        raise NotImplementedError(
            "AskUserQuestion tool requires async execution. Use _arun instead."
        )
    
    async def _arun(self, questions: list[Question]) -> dict[str, Any]:
        """Ask user questions and wait for responses.
        
        Args:
            questions: List of questions to ask the user
            
        Returns:
            Dictionary with 'answers' (list of Answer objects) and 'cancelled' (bool)
            
        Raises:
            RuntimeError: If user_input_callback is not set
        """
        if self.user_input_callback is None:
            logger.error("AskUserQuestion tool called without user_input_callback set")
            return {
                "answers": [],
                "cancelled": True,
                "error": "User input is not available in this context"
            }
        
        logger.info(f"Asking user {len(questions)} question(s)")
        
        try:
            result = await self.user_input_callback(questions)
            
            if result.cancelled:
                logger.info("User cancelled the questions")
            else:
                logger.info(f"User answered {len(result.answers)} question(s)")
                for answer in result.answers:
                    logger.debug(f"Q: {answer.question[:50]}... A: {answer.answer}")
            
            return result.model_dump()
            
        except Exception as e:
            logger.error(f"Error getting user input: {e}", exc_info=True)
            return {
                "answers": [],
                "cancelled": True,
                "error": f"Failed to get user input: {str(e)}"
            }


def create_ask_user_question_tool(
    callback: Optional[UserInputCallback] = None
) -> AskUserQuestionTool:
    """Create an AskUserQuestion tool with a callback.
    
    Args:
        callback: Async function to handle user input requests
        
    Returns:
        Configured AskUserQuestionTool instance
        
    Example:
        >>> async def my_callback(questions):
        ...     # Show questions in UI and get answers
        ...     return AskUserQuestionResult(answers=[...])
        >>> tool = create_ask_user_question_tool(my_callback)
    """
    tool = AskUserQuestionTool()
    if callback:
        tool.user_input_callback = callback
    return tool


# Synchronous version for simple CLI use
def ask_user_simple(question: str, options: list[str]) -> Optional[str]:
    """Simple synchronous version for basic CLI usage.
    
    Args:
        question: Question to ask
        options: List of option labels
        
    Returns:
        Selected option or custom answer, None if cancelled
    """
    print(f"\n{question}")
    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")
    print(f"  {len(options) + 1}. Other (custom answer)")
    print(f"  0. Cancel")
    
    while True:
        try:
            choice = input("\nYour choice: ").strip()
            if not choice:
                continue
            
            choice_num = int(choice)
            
            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(options):
                return options[choice_num - 1]
            elif choice_num == len(options) + 1:
                custom = input("Enter your answer: ").strip()
                return custom if custom else None
            else:
                print(f"Please enter a number between 0 and {len(options) + 1}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            return None


__all__ = [
    "Choice",
    "Question",
    "Answer",
    "AskUserQuestionInput",
    "AskUserQuestionResult",
    "AskUserQuestionTool",
    "UserInputCallback",
    "create_ask_user_question_tool",
    "ask_user_simple",
]
