"""Example: Using the AskUserQuestion tool with ChATLAS agents.

This example demonstrates how to integrate the AskUserQuestion tool
into your agent workflow for interactive user input.
"""

import asyncio
from chatlas_agents.tools import (
    create_ask_user_question_tool,
    Question,
    Choice,
    AskUserQuestionResult,
    Answer,
)
from deepagents import create_deep_agent


async def simple_callback(questions: list[Question]) -> AskUserQuestionResult:
    """Simple CLI callback for demonstration purposes.
    
    In a real application, this would show a UI with tabs, radio buttons, etc.
    """
    print("\n" + "="*70)
    print("AGENT NEEDS YOUR INPUT")
    print("="*70)
    
    answers = []
    
    for i, q in enumerate(questions, 1):
        print(f"\n[Question {i}/{len(questions)}]")
        if q.header:
            print(f"Category: {q.header}")
        print(f"\n{q.question}\n")
        
        # Show options
        for j, opt in enumerate(q.options, 1):
            print(f"  {j}. {opt.label}")
            if opt.description:
                print(f"     {opt.description}")
        
        print(f"  {len(q.options) + 1}. Other (custom answer)")
        print(f"  0. Cancel")
        
        # Get user input
        while True:
            try:
                choice = input("\nYour choice: ").strip()
                if not choice:
                    continue
                
                choice_num = int(choice)
                
                if choice_num == 0:
                    return AskUserQuestionResult(answers=[], cancelled=True)
                elif 1 <= choice_num <= len(q.options):
                    answer = q.options[choice_num - 1].label
                    answers.append(Answer(
                        question=q.question,
                        answer=answer,
                        is_other=False
                    ))
                    break
                elif choice_num == len(q.options) + 1:
                    custom = input("Enter your answer: ").strip()
                    if custom:
                        answers.append(Answer(
                            question=q.question,
                            answer=custom,
                            is_other=True
                        ))
                        break
                else:
                    print(f"Please enter 0-{len(q.options) + 1}")
            except (ValueError, KeyboardInterrupt, EOFError):
                return AskUserQuestionResult(answers=[], cancelled=True)
    
    print("\n" + "="*70 + "\n")
    return AskUserQuestionResult(answers=answers, cancelled=False)


async def main():
    """Example usage of AskUserQuestion tool."""
    
    # Create the tool with our callback
    ask_tool = create_ask_user_question_tool(simple_callback)
    
    # Create agent with the tool
    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-5-20250929",
        tools=[ask_tool],
        system_prompt="""You are a helpful assistant for ATLAS physicists.
        
When you need to clarify requirements or gather user preferences, 
use the ask_user_question tool to ask interactive questions.

For example:
- When asked to analyze data, ask about the preferred analysis framework
- When creating documentation, ask about the target audience
- When optimizing code, ask about priorities (speed vs readability)
"""
    )
    
    # Example 1: Agent asks about analysis preferences
    print("="*70)
    print("EXAMPLE 1: Agent asking about analysis preferences")
    print("="*70)
    
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": "I need help setting up an ATLAS analysis. Can you help?"
        }]
    })
    
    # The agent might use ask_user_question to clarify:
    # - Which analysis framework? (Athena, AnalysisBase, etc.)
    # - Data type? (MC, Data, Both)
    # - Analysis type? (Cut-based, MVA, etc.)
    
    print("\nAgent response:")
    for msg in result["messages"]:
        if msg.type == "ai":
            print(msg.content)
    
    print("\n" + "="*70)


async def example_direct_usage():
    """Direct usage of the tool without an agent."""
    
    print("="*70)
    print("EXAMPLE 2: Direct tool usage")
    print("="*70)
    
    # Create tool
    ask_tool = create_ask_user_question_tool(simple_callback)
    
    # Define questions
    questions = [
        Question(
            question="What's your main goal for this refactoring?",
            header="Goal",
            options=[
                Choice(
                    label="Performance",
                    description="Make the code run faster"
                ),
                Choice(
                    label="Readability",
                    description="Make the code easier to understand"
                ),
                Choice(
                    label="Maintainability",
                    description="Make the code easier to modify in the future"
                ),
            ]
        ),
        Question(
            question="Which Python version should we target?",
            header="Python",
            options=[
                Choice(label="Python 3.11+"),
                Choice(label="Python 3.12+"),
                Choice(label="Python 3.13+"),
            ]
        ),
    ]
    
    # Ask the questions
    result = await ask_tool._arun(questions)
    
    print("\nResults:")
    if result["cancelled"]:
        print("  User cancelled")
    else:
        for answer in result["answers"]:
            print(f"  Q: {answer['question'][:50]}...")
            print(f"  A: {answer['answer']}")
            if answer.get("is_other"):
                print(f"     (custom answer)")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import os
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Skipping agent example.")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'\n")
        
        # Just run direct usage example
        asyncio.run(example_direct_usage())
    else:
        # Run full example with agent
        asyncio.run(main())
        asyncio.run(example_direct_usage())
