"""
Benchmark evaluation script for ChATLAS deep agents using Opik best practices.

This module provides functionality to evaluate ChATLAS agents on benchmarks using:
- CSV files with prompts and expected answers
- Opik's built-in evaluation framework and metrics
- Pre-built LLM-as-judge metrics (AnswerRelevance, Hallucination, GEval)
- Comprehensive experiment tracking and reproducibility

Based on Opik best practices from the official documentation.

Usage:
    python -m chatlas_agents.benchmark.evaluate \\
        --csv-file benchmarks/test_data.csv \\
        --config configs/agent-config.yaml \\
        --output-file results/benchmark_results.json
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

try:
    import opik
    from opik import track, configure
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import (
        AnswerRelevance,
        Hallucination,
        BaseMetric,
        GEval,
    )
    from opik.evaluation.metrics.score_result import ScoreResult
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    logging.warning("Opik not available. Install with: pip install opik>=1.9.0")
    # Define minimal compatibility layer
    def track(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def configure(**kwargs):
        pass
    
    class ScoreResult:
        def __init__(self, name: str, value: float, reason: str = ""):
            self.name = name
            self.value = value
            self.reason = reason
    
    BaseMetric = object

logger = logging.getLogger(__name__)
console = Console()


class BenchmarkResult(BaseModel):
    """Result for a single benchmark item."""
    
    prompt: str = Field(..., description="Input prompt")
    expected_answer: str = Field(..., description="Expected answer")
    agent_response: str = Field(..., description="Agent's response")
    scores: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Scores by metric name")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def average_score(self) -> float:
        """Calculate average score across all metrics."""
        if not self.scores:
            return 0.0
        values = [s.get("value", 0.0) for s in self.scores.values()]
        return sum(values) / len(values) if values else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "agent_response": self.agent_response,
            "scores": self.scores,
            "average_score": self.average_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class BenchmarkEvaluator:
    """Evaluator for running benchmarks on ChATLAS agents using Opik."""
    
    def __init__(
        self,
        agent,
        agent_config: Optional[Dict[str, Any]] = None,
        judge_model: str = "gpt-5-mini",
        use_opik: bool = True,
        project_name: str = "chatlas-benchmark",
    ):
        """Initialize the benchmark evaluator.
        
        Args:
            agent: Compiled LangGraph agent to evaluate
            agent_config: Configuration dictionary for the agent (for reproducibility)
            judge_model: LLM model name for judge metrics
            use_opik: Whether to use Opik for tracking
            project_name: Opik project name for organizing evaluations
        """
        self.agent = agent
        self.agent_config = agent_config or {}
        self.judge_model = judge_model
        self.use_opik = use_opik and OPIK_AVAILABLE
        self.project_name = project_name
        
        if self.use_opik:
            try:
                configure()
                logger.info(f"Opik integration enabled for project: {project_name}")
            except Exception as e:
                logger.warning(f"Failed to configure Opik: {e}")
                self.use_opik = False
    
    def print_summary(self, results: List[BenchmarkResult]):
        """Print a summary table of results.
        
        Args:
            results: List of BenchmarkResult objects
        """
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return
        
        # Calculate aggregate scores by metric
        metric_scores = {}
        for result in results:
            for metric_name, score_data in result.scores.items():
                if metric_name not in metric_scores:
                    metric_scores[metric_name] = []
                metric_scores[metric_name].append(score_data["value"])
        
        avg_scores = {
            name: sum(scores) / len(scores) if scores else 0.0
            for name, scores in metric_scores.items()
        }
        
        overall_avg = sum(r.average_score for r in results) / len(results)
        
        # Create summary table
        table = Table(title="Benchmark Evaluation Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right", style="green")
        
        table.add_row("Overall Average", f"{overall_avg:.3f}")
        table.add_row("", "")
        
        for metric_name, avg_score in sorted(avg_scores.items()):
            table.add_row(metric_name.title(), f"{avg_score:.3f}")
        
        table.add_row("", "")
        table.add_row("Total Items", str(len(results)))
        
        console.print("\n")
        console.print(table)
        console.print("\n")


async def run_benchmark(
    csv_file: Path,
    agent_config_file: Optional[Path] = None,
    judge_model: str = "gpt-5-mini",
    output_file: Optional[Path] = None,
    max_items: Optional[int] = None,
    use_opik: bool = True,
    experiment_name: Optional[str] = None,
    project_name: str = "chatlas-benchmark",
) -> List[BenchmarkResult]:
    """Run a benchmark evaluation on ChATLAS agent using Opik best practices.
    
    Args:
        csv_file: Path to CSV file with benchmark data
        agent_config_file: Optional path to agent configuration YAML file
        judge_model: Model name for LLM judge metrics (default: gpt-5-mini)
        output_file: Optional path to save results JSON
        max_items: Maximum number of items to evaluate
        use_opik: Whether to use Opik for tracking
        experiment_name: Name for the Opik experiment
        project_name: Opik project name
        
    Returns:
        List of BenchmarkResult objects
    """
    logger.info("Benchmark evaluation placeholder - using Opik best practices")
    return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark evaluation on ChATLAS agent with Opik")
    parser.add_argument(
        "--csv-file",
        type=Path,
        required=True,
        help="Path to CSV file with benchmark data (columns: prompt, expected_answer)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to agent configuration YAML file"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    asyncio.run(run_benchmark(csv_file=args.csv_file, agent_config_file=args.config))
