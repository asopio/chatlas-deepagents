"""
Simple test script for the benchmark evaluation module.

This script creates a minimal benchmark and runs a basic evaluation
to verify that the core functionality works.
"""

import asyncio
import json
import tempfile
from pathlib import Path

from chatlas_agents.benchmark import BenchmarkResult, CategoryScore


def test_category_score():
    """Test CategoryScore model."""
    score = CategoryScore(
        category="accuracy",
        score=4,
        reasoning="Response is mostly accurate with minor issues"
    )
    assert score.category == "accuracy"
    assert score.score == 4
    assert 1 <= score.score <= 5
    print("✓ CategoryScore test passed")


def test_benchmark_result():
    """Test BenchmarkResult model."""
    scores = [
        CategoryScore(category="accuracy", score=4, reasoning="Good accuracy"),
        CategoryScore(category="relevance", score=5, reasoning="Very relevant"),
        CategoryScore(category="coverage", score=3, reasoning="Partial coverage"),
        CategoryScore(category="conciseness", score=4, reasoning="Concise enough"),
    ]
    
    result = BenchmarkResult(
        prompt="What is ATLAS?",
        expected_answer="ATLAS is a particle physics experiment",
        agent_response="ATLAS is a detector at CERN's LHC",
        scores=scores,
        metadata={"category": "physics", "difficulty": "easy"}
    )
    
    assert result.prompt == "What is ATLAS?"
    assert len(result.scores) == 4
    assert result.average_score == 4.0  # (4+5+3+4)/4
    
    # Test serialization
    result_dict = result.to_dict()
    assert "prompt" in result_dict
    assert "average_score" in result_dict
    assert result_dict["average_score"] == 4.0
    
    print("✓ BenchmarkResult test passed")


def test_csv_format():
    """Test CSV file format validation."""
    import pandas as pd
    
    # Create a test CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("prompt,expected_answer,category\n")
        f.write("What is ATLAS?,ATLAS is a detector,physics\n")
        f.write("How to use Rucio?,Rucio is for data management,technical\n")
        csv_path = Path(f.name)
    
    try:
        # Read and validate
        df = pd.read_csv(csv_path)
        assert "prompt" in df.columns
        assert "expected_answer" in df.columns
        assert len(df) == 2
        assert df.iloc[0]["category"] == "physics"
        print("✓ CSV format test passed")
    finally:
        csv_path.unlink()


async def test_llm_judge_parsing():
    """Test that we can parse LLM judge responses correctly."""
    # Simulate a judge response in JSON format
    mock_response = """
    {
      "accuracy": {"score": 4, "reasoning": "Response is accurate"},
      "relevance": {"score": 5, "reasoning": "Highly relevant"},
      "coverage": {"score": 3, "reasoning": "Covers main points"},
      "conciseness": {"score": 4, "reasoning": "Reasonably concise"}
    }
    """
    
    # Parse it
    scores_dict = json.loads(mock_response.strip())
    
    # Convert to CategoryScore objects
    scores = []
    for category in ["accuracy", "relevance", "coverage", "conciseness"]:
        if category in scores_dict:
            category_data = scores_dict[category]
            scores.append(CategoryScore(
                category=category,
                score=category_data["score"],
                reasoning=category_data["reasoning"]
            ))
    
    assert len(scores) == 4
    assert all(1 <= s.score <= 5 for s in scores)
    assert scores[0].category == "accuracy"
    assert scores[0].score == 4
    
    print("✓ LLM judge parsing test passed")


async def test_json_extraction():
    """Test extracting JSON from markdown code blocks."""
    # Test with markdown code block
    mock_response_with_markdown = """
    Here's my evaluation:
    
    ```json
    {
      "accuracy": {"score": 4, "reasoning": "Good"},
      "relevance": {"score": 5, "reasoning": "Excellent"},
      "coverage": {"score": 3, "reasoning": "Partial"},
      "conciseness": {"score": 4, "reasoning": "Clear"}
    }
    ```
    
    That's my assessment.
    """
    
    # Extract JSON
    response_text = mock_response_with_markdown
    if "```json" in response_text:
        json_start = response_text.find("```json") + 7
        json_end = response_text.find("```", json_start)
        response_text = response_text[json_start:json_end].strip()
    
    # Parse it
    scores_dict = json.loads(response_text)
    assert "accuracy" in scores_dict
    assert scores_dict["relevance"]["score"] == 5
    
    print("✓ JSON extraction test passed")


def main():
    """Run all tests."""
    print("Running benchmark module tests...\n")
    
    try:
        # Synchronous tests
        test_category_score()
        test_benchmark_result()
        test_csv_format()
        
        # Async tests
        asyncio.run(test_llm_judge_parsing())
        asyncio.run(test_json_extraction())
        
        print("\n✅ All tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
