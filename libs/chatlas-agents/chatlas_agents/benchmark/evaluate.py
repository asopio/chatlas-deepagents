"""
Benchmark evaluation script for ChATLAS deep agents.

This module provides functionality to evaluate ChATLAS agents on benchmarks using:
- CSV files with prompts and expected answers
- LLM-as-judge scoring (accuracy, relevance, coverage, conciseness)
- Opik integration for logging and reproducibility

Usage:
    python -m chatlas_agents.benchmark.evaluate \\
        --csv-file benchmarks/test_data.csv \\
        --config configs/agent-config.yaml \\
        --judge-model gpt-5-mini \\
        --output-file results/benchmark_results.json
"""

import asyncio
import csv
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

try:
    import opik
    from opik import track
    from opik.evaluation import evaluate
    from opik.evaluation.metrics import base_metric
    OPIK_AVAILABLE = True
except ImportError:
    OPIK_AVAILABLE = False
    logging.warning("Opik not available. Install with: pip install opik")
    # Define a no-op decorator when Opik is not available
    def track(name=None, **kwargs):
        def decorator(func):
            return func
        return decorator

logger = logging.getLogger(__name__)
console = Console()


class CategoryScore(BaseModel):
    """Score for a single evaluation category."""
    
    category: str = Field(..., description="Category name (accuracy, relevance, coverage, conciseness)")
    score: int = Field(..., ge=1, le=5, description="Score from 1 to 5")
    reasoning: str = Field(..., description="Explanation for the score")


class BenchmarkResult(BaseModel):
    """Result for a single benchmark item."""
    
    prompt: str = Field(..., description="Input prompt")
    expected_answer: str = Field(..., description="Expected answer")
    agent_response: str = Field(..., description="Agent's response")
    scores: List[CategoryScore] = Field(..., description="Scores for each category")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def average_score(self) -> float:
        """Calculate average score across all categories."""
        if not self.scores:
            return 0.0
        return sum(s.score for s in self.scores) / len(self.scores)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "agent_response": self.agent_response,
            "scores": [s.model_dump() for s in self.scores],
            "average_score": self.average_score,
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class LLMJudge:
    """LLM-as-judge for scoring agent responses."""
    
    JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for AI agent responses.
You will be given:
1. A user prompt
2. The expected answer
3. The agent's actual response

Your task is to evaluate the agent's response on the following dimensions:
1. ACCURACY: Is the response factually correct and aligned with the expected answer?
2. RELEVANCE: Does the response directly address the user's prompt?
3. COVERAGE: Does the response cover all important aspects of the expected answer?
4. CONCISENESS: Is the response clear and concise without unnecessary information?

For each dimension, provide:
- A score from 1 to 5 (1=very poor, 2=poor, 3=adequate, 4=good, 5=excellent)
- A brief explanation (1-2 sentences) for your score

Respond in the following JSON format:
{
  "accuracy": {"score": <1-5>, "reasoning": "<explanation>"},
  "relevance": {"score": <1-5>, "reasoning": "<explanation>"},
  "coverage": {"score": <1-5>, "reasoning": "<explanation>"},
  "conciseness": {"score": <1-5>, "reasoning": "<explanation>"}
}
"""
    
    def __init__(self, judge_model: BaseChatModel):
        """Initialize the LLM judge.
        
        Args:
            judge_model: LangChain chat model to use for judging
        """
        self.judge_model = judge_model
    
    async def evaluate_response(
        self,
        prompt: str,
        expected_answer: str,
        agent_response: str
    ) -> List[CategoryScore]:
        """Evaluate an agent response using LLM-as-judge.
        
        Args:
            prompt: Original user prompt
            expected_answer: Expected answer
            agent_response: Agent's actual response
            
        Returns:
            List of CategoryScore objects for each evaluation dimension
        """
        evaluation_prompt = f"""
USER PROMPT:
{prompt}

EXPECTED ANSWER:
{expected_answer}

AGENT'S RESPONSE:
{agent_response}

Please evaluate the agent's response on the four dimensions: accuracy, relevance, coverage, and conciseness.
Provide your evaluation in the specified JSON format.
"""
        
        messages = [
            SystemMessage(content=self.JUDGE_SYSTEM_PROMPT),
            HumanMessage(content=evaluation_prompt)
        ]
        
        try:
            response = await self.judge_model.ainvoke(messages)
            response_text = response.content
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            scores_dict = json.loads(response_text)
            
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
            
            return scores
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            # Return default scores on error
            return [
                CategoryScore(category="accuracy", score=1, reasoning=f"Error during evaluation: {e}"),
                CategoryScore(category="relevance", score=1, reasoning=f"Error during evaluation: {e}"),
                CategoryScore(category="coverage", score=1, reasoning=f"Error during evaluation: {e}"),
                CategoryScore(category="conciseness", score=1, reasoning=f"Error during evaluation: {e}"),
            ]


class BenchmarkEvaluator:
    """Evaluator for running benchmarks on ChATLAS agents."""
    
    def __init__(
        self,
        agent,
        judge_model: BaseChatModel,
        agent_config: Optional[Dict[str, Any]] = None,
        use_opik: bool = True,
    ):
        """Initialize the benchmark evaluator.
        
        Args:
            agent: Compiled LangGraph agent to evaluate
            judge_model: LangChain chat model for LLM-as-judge
            agent_config: Configuration dictionary for the agent (for reproducibility)
            use_opik: Whether to use Opik for logging and tracking
        """
        self.agent = agent
        self.judge = LLMJudge(judge_model)
        self.agent_config = agent_config or {}
        self.use_opik = use_opik and OPIK_AVAILABLE
        
        if self.use_opik:
            try:
                # Initialize Opik client
                opik.configure()
                logger.info("Opik integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Opik: {e}")
                self.use_opik = False
    
    async def run_agent_on_prompt(
        self,
        prompt: str,
        thread_id: Optional[str] = None
    ) -> str:
        """Run the agent on a single prompt.
        
        Args:
            prompt: Input prompt
            thread_id: Optional thread ID for conversation context
            
        Returns:
            Agent's response as a string
        """
        config = {"configurable": {"thread_id": thread_id or "benchmark"}}
        
        try:
            result = await self.agent.ainvoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config
            )
            
            # Extract the final response from the agent
            messages = result.get("messages", [])
            if messages:
                last_message = messages[-1]
                if hasattr(last_message, "content"):
                    return last_message.content
                elif isinstance(last_message, dict):
                    return last_message.get("content", "")
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return f"ERROR: {e}"
    
    @track(name="evaluate_benchmark_item")
    async def evaluate_single_item(
        self,
        prompt: str,
        expected_answer: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkResult:
        """Evaluate agent on a single benchmark item.
        
        Args:
            prompt: Input prompt
            expected_answer: Expected answer
            metadata: Optional metadata about the item
            
        Returns:
            BenchmarkResult with scores and details
        """
        # Run agent
        agent_response = await self.run_agent_on_prompt(prompt)
        
        # Evaluate with LLM judge
        scores = await self.judge.evaluate_response(
            prompt=prompt,
            expected_answer=expected_answer,
            agent_response=agent_response
        )
        
        result = BenchmarkResult(
            prompt=prompt,
            expected_answer=expected_answer,
            agent_response=agent_response,
            scores=scores,
            metadata=metadata or {}
        )
        
        # Log to Opik if enabled
        if self.use_opik:
            try:
                opik.log_traces(
                    name="benchmark_evaluation",
                    input={"prompt": prompt, "expected": expected_answer},
                    output={"response": agent_response, "scores": [s.model_dump() for s in scores]},
                    metadata={**self.agent_config, **(metadata or {})},
                    tags=["benchmark", "evaluation"],
                )
            except Exception as e:
                logger.warning(f"Failed to log to Opik: {e}")
        
        return result
    
    async def evaluate_from_csv(
        self,
        csv_file: Path,
        max_items: Optional[int] = None
    ) -> List[BenchmarkResult]:
        """Evaluate agent on benchmark items from a CSV file.
        
        Args:
            csv_file: Path to CSV file with columns: prompt, expected_answer, (optional: metadata columns)
            max_items: Maximum number of items to evaluate (None = all)
            
        Returns:
            List of BenchmarkResult objects
        """
        # Load CSV
        df = pd.read_csv(csv_file)
        
        if "prompt" not in df.columns or "expected_answer" not in df.columns:
            raise ValueError("CSV must have 'prompt' and 'expected_answer' columns")
        
        # Limit items if specified
        if max_items:
            df = df.head(max_items)
        
        results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                f"[cyan]Evaluating {len(df)} benchmark items...",
                total=len(df)
            )
            
            for idx, row in df.iterrows():
                prompt = row["prompt"]
                expected_answer = row["expected_answer"]
                
                # Extract metadata from other columns
                metadata = {
                    col: row[col]
                    for col in df.columns
                    if col not in ["prompt", "expected_answer"] and pd.notna(row[col])
                }
                metadata["csv_row"] = int(idx)
                
                console.print(f"\n[bold blue]Evaluating item {idx + 1}/{len(df)}[/bold blue]")
                console.print(f"[dim]Prompt: {prompt[:100]}...[/dim]")
                
                result = await self.evaluate_single_item(
                    prompt=prompt,
                    expected_answer=expected_answer,
                    metadata=metadata
                )
                
                results.append(result)
                
                # Show scores
                avg_score = result.average_score
                console.print(f"[green]Average score: {avg_score:.2f}/5.0[/green]")
                
                progress.update(task, advance=1)
        
        return results
    
    def save_results(
        self,
        results: List[BenchmarkResult],
        output_file: Path
    ):
        """Save benchmark results to a JSON file.
        
        Args:
            results: List of BenchmarkResult objects
            output_file: Path to output JSON file
        """
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "agent_config": self.agent_config,
            "num_items": len(results),
            "average_score": sum(r.average_score for r in results) / len(results) if results else 0.0,
            "results": [r.to_dict() for r in results]
        }
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"\n[green]✓ Results saved to {output_file}[/green]")
    
    def print_summary(self, results: List[BenchmarkResult]):
        """Print a summary table of results.
        
        Args:
            results: List of BenchmarkResult objects
        """
        if not results:
            console.print("[yellow]No results to display[/yellow]")
            return
        
        # Calculate aggregate scores
        avg_scores_by_category = {}
        for category in ["accuracy", "relevance", "coverage", "conciseness"]:
            category_scores = []
            for result in results:
                for score in result.scores:
                    if score.category == category:
                        category_scores.append(score.score)
            avg_scores_by_category[category] = (
                sum(category_scores) / len(category_scores) if category_scores else 0.0
            )
        
        overall_avg = sum(r.average_score for r in results) / len(results)
        
        # Create summary table
        table = Table(title="Benchmark Evaluation Summary", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right", style="green")
        
        table.add_row("Overall Average", f"{overall_avg:.2f}/5.0")
        table.add_row("", "")
        table.add_row("Accuracy", f"{avg_scores_by_category['accuracy']:.2f}/5.0")
        table.add_row("Relevance", f"{avg_scores_by_category['relevance']:.2f}/5.0")
        table.add_row("Coverage", f"{avg_scores_by_category['coverage']:.2f}/5.0")
        table.add_row("Conciseness", f"{avg_scores_by_category['conciseness']:.2f}/5.0")
        table.add_row("", "")
        table.add_row("Total Items", str(len(results)))
        
        console.print("\n")
        console.print(table)
        console.print("\n")


async def run_benchmark(
    csv_file: Path,
    agent_config_file: Optional[Path] = None,
    judge_model_name: str = "gpt-5-mini",
    judge_provider: str = "openai",
    output_file: Optional[Path] = None,
    max_items: Optional[int] = None,
    use_opik: bool = True,
) -> List[BenchmarkResult]:
    """Run a benchmark evaluation on ChATLAS agent.
    
    Args:
        csv_file: Path to CSV file with benchmark data
        agent_config_file: Optional path to agent configuration YAML file
        judge_model_name: Model name for LLM judge (default: gpt-5-mini)
        judge_provider: LLM provider for judge (default: openai)
        output_file: Optional path to save results JSON
        max_items: Maximum number of items to evaluate
        use_opik: Whether to use Opik for tracking
        
    Returns:
        List of BenchmarkResult objects
    """
    from chatlas_agents.config import load_config_from_yaml, load_config_from_env
    from chatlas_agents.llm import create_llm_from_config
    from chatlas_agents.middleware import MCPMiddleware
    from deepagents import create_deep_agent
    
    logger.info("Starting benchmark evaluation...")
    
    # Load agent configuration
    if agent_config_file:
        config = load_config_from_yaml(str(agent_config_file))
        logger.info(f"Loaded agent config from {agent_config_file}")
    else:
        config = load_config_from_env()
        logger.info("Loaded agent config from environment")
    
    agent_config_dict = config.model_dump()
    
    # Create LLM for agent
    agent_llm = create_llm_from_config(config.llm)
    
    # Create MCP middleware if configured
    middleware = []
    if config.mcp.url:
        try:
            logger.info(f"Loading MCP tools from {config.mcp.url}...")
            mcp_middleware = await MCPMiddleware.create(config.mcp)
            middleware.append(mcp_middleware)
            logger.info(f"Loaded {len(mcp_middleware.tools)} MCP tools")
        except Exception as e:
            logger.warning(f"Failed to load MCP middleware: {e}")
    
    # Create agent
    agent = create_deep_agent(
        model=agent_llm,
        middleware=middleware,
        system_prompt="You are a helpful AI assistant for ATLAS experiment documentation and queries.",
    )
    
    # Create judge LLM
    from chatlas_agents.config import LLMConfig, LLMProvider
    judge_config = LLMConfig(
        provider=LLMProvider(judge_provider),
        model=judge_model_name,
    )
    judge_llm = create_llm_from_config(judge_config)
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(
        agent=agent,
        judge_model=judge_llm,
        agent_config=agent_config_dict,
        use_opik=use_opik,
    )
    
    # Run evaluation
    results = await evaluator.evaluate_from_csv(csv_file, max_items=max_items)
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    if output_file:
        evaluator.save_results(results, output_file)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run benchmark evaluation on ChATLAS agent")
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
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5-mini",
        help="Model name for LLM judge (default: gpt-5-mini)"
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default="openai",
        choices=["openai", "anthropic", "groq"],
        help="LLM provider for judge (default: openai)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to save results JSON file"
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Maximum number of items to evaluate"
    )
    parser.add_argument(
        "--no-opik",
        action="store_true",
        help="Disable Opik tracking"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run benchmark
    asyncio.run(
        run_benchmark(
            csv_file=args.csv_file,
            agent_config_file=args.config,
            judge_model_name=args.judge_model,
            judge_provider=args.judge_provider,
            output_file=args.output,
            max_items=args.max_items,
            use_opik=not args.no_opik,
        )
    )
