"""
Integration test for the benchmark evaluation framework.

This test creates a simple mock agent and runs a benchmark evaluation
to verify the end-to-end pipeline works correctly.
"""

import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock


async def test_benchmark_evaluation_integration():
    """Test complete benchmark evaluation pipeline with mock agent."""
    from chatlas_agents.benchmark.evaluate import (
        BenchmarkEvaluator,
        LLMJudge,
        CategoryScore,
    )
    
    # Create a temporary CSV file
    csv_content = """prompt,expected_answer,category
"What is ATLAS?","ATLAS is a particle detector",physics
"How to use Rucio?","Rucio is for data management",technical
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        csv_path = Path(f.name)
    
    try:
        # Create mock agent
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value={
            "messages": [
                MagicMock(content="ATLAS is a large detector at CERN's LHC for studying particle physics.")
            ]
        })
        
        # Create mock judge model
        mock_judge = AsyncMock()
        mock_judge_response = MagicMock()
        mock_judge_response.content = """
        {
          "accuracy": {"score": 4, "reasoning": "Response is accurate"},
          "relevance": {"score": 5, "reasoning": "Directly relevant"},
          "coverage": {"score": 4, "reasoning": "Covers main points"},
          "conciseness": {"score": 3, "reasoning": "Could be more concise"}
        }
        """
        mock_judge.ainvoke = AsyncMock(return_value=mock_judge_response)
        
        # Create evaluator
        evaluator = BenchmarkEvaluator(
            agent=mock_agent,
            judge_model=mock_judge,
            agent_config={"model": "test-model"},
            use_opik=False,  # Disable Opik for testing
        )
        
        # Run evaluation on a single item
        result = await evaluator.evaluate_single_item(
            prompt="What is ATLAS?",
            expected_answer="ATLAS is a particle detector",
            metadata={"category": "physics"}
        )
        
        # Verify result structure
        assert result.prompt == "What is ATLAS?"
        assert result.expected_answer == "ATLAS is a particle detector"
        assert "ATLAS" in result.agent_response
        assert len(result.scores) == 4
        
        # Verify scores
        score_categories = {s.category for s in result.scores}
        assert score_categories == {"accuracy", "relevance", "coverage", "conciseness"}
        
        # Verify average score calculation
        expected_avg = (4 + 5 + 4 + 3) / 4  # 4.0
        assert result.average_score == expected_avg
        
        # Test save results
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = Path(f.name)
        
        try:
            evaluator.save_results([result], output_path)
            
            # Verify output file
            with open(output_path) as f:
                saved_data = json.load(f)
            
            assert "results" in saved_data
            assert len(saved_data["results"]) == 1
            assert saved_data["average_score"] == expected_avg
            assert saved_data["num_items"] == 1
            
        finally:
            output_path.unlink()
        
        print("✓ Integration test passed")
        
    finally:
        csv_path.unlink()


async def test_llm_judge_error_handling():
    """Test that LLM judge handles errors gracefully."""
    from chatlas_agents.benchmark.evaluate import LLMJudge
    from unittest.mock import AsyncMock
    
    # Create mock judge that raises an error
    mock_judge = AsyncMock()
    mock_judge.ainvoke = AsyncMock(side_effect=Exception("Mock error"))
    
    judge = LLMJudge(mock_judge)
    
    # Should return default scores on error
    scores = await judge.evaluate_response(
        prompt="Test prompt",
        expected_answer="Test answer",
        agent_response="Test response"
    )
    
    assert len(scores) == 4
    assert all(s.score == 1 for s in scores)
    assert all("Error during evaluation" in s.reasoning for s in scores)
    
    print("✓ Error handling test passed")


if __name__ == "__main__":
    asyncio.run(test_benchmark_evaluation_integration())
    asyncio.run(test_llm_judge_error_handling())
    print("\n✅ All integration tests passed!")
