# ChATLAS Agent Benchmark Evaluation

This module provides a comprehensive evaluation framework for ChATLAS deep agents using:
- **CSV-based benchmarks** with prompts and expected answers
- **LLM-as-judge scoring** across 4 dimensions: accuracy, relevance, coverage, conciseness
- **Opik integration** for logging, monitoring, and ensuring reproducibility

## Features

### 1. LLM-as-Judge Scoring
Each agent response is evaluated by a separate LLM (the "judge") on four key dimensions:

- **Accuracy** (1-5): Is the response factually correct and aligned with the expected answer?
- **Relevance** (1-5): Does the response directly address the user's prompt?
- **Coverage** (1-5): Does the response cover all important aspects of the expected answer?
- **Conciseness** (1-5): Is the response clear and concise without unnecessary information?

### 2. Opik Integration
[Opik](https://www.comet.com/docs/opik/) is used for:
- Logging all agent interactions and evaluations
- Tracking model configurations for reproducibility
- Monitoring benchmark performance over time
- Comparing different agent configurations

### 3. CSV Benchmark Format
Benchmarks are defined in CSV files with the following columns:
- `prompt`: The input question/prompt for the agent
- `expected_answer`: The ground-truth answer
- Additional metadata columns (e.g., `category`, `difficulty`)

## Installation

Install the required dependencies:

```bash
cd libs/chatlas-agents
pip install -r requirements.txt

# Or with uv
uv pip install -r requirements.txt
```

This will install:
- `opik>=0.2.0` - For evaluation tracking and monitoring
- `pandas>=2.0.0` - For CSV processing
- All ChATLAS agent dependencies

## Quick Start

### 1. Prepare Your Benchmark Data

Create a CSV file with your benchmark prompts and expected answers:

```csv
prompt,expected_answer,category,difficulty
"What is the ATLAS experiment?","The ATLAS experiment is a particle physics detector at CERN's LHC...",physics_general,easy
"How do I query datasets in AMI?","AMI can be queried using pyAMI with commands like...",technical,medium
```

See `benchmarks/example_benchmark.csv` for a complete example.

### 2. Set Up Environment Variables

Configure your API keys for the agent and judge models:

```bash
# Agent configuration
export CHATLAS_LLM_PROVIDER=openai
export CHATLAS_LLM_MODEL=gpt-5-mini
export OPENAI_API_KEY=your-openai-key

# MCP server (optional)
export CHATLAS_MCP_URL=https://chatlas-mcp.app.cern.ch/mcp
export CHATLAS_MCP_TIMEOUT=120

# Opik configuration (optional)
export OPIK_API_KEY=your-opik-key
export OPIK_WORKSPACE=your-workspace
```

### 3. Run the Benchmark

```bash
python -m chatlas_agents.benchmark.evaluate \
  --csv-file benchmarks/example_benchmark.csv \
  --judge-model gpt-5-mini \
  --judge-provider openai \
  --output results/benchmark_results.json \
  --verbose
```

## Usage Examples

### Basic Evaluation

Evaluate your agent on a benchmark CSV:

```bash
python -m chatlas_agents.benchmark.evaluate \
  --csv-file benchmarks/my_benchmark.csv \
  --output results/results.json
```

### Custom Agent Configuration

Use a YAML configuration file for the agent:

```bash
python -m chatlas_agents.benchmark.evaluate \
  --csv-file benchmarks/my_benchmark.csv \
  --config configs/my-agent-config.yaml \
  --output results/results.json
```

### Custom Judge Model

Use a different model as the judge:

```bash
python -m chatlas_agents.benchmark.evaluate \
  --csv-file benchmarks/my_benchmark.csv \
  --judge-model claude-3-5-sonnet-20241022 \
  --judge-provider anthropic \
  --output results/results.json
```

### Limit Number of Items

Test on a subset of items first:

```bash
python -m chatlas_agents.benchmark.evaluate \
  --csv-file benchmarks/my_benchmark.csv \
  --max-items 5 \
  --output results/test_results.json
```

### Disable Opik Tracking

Run without Opik integration:

```bash
python -m chatlas_agents.benchmark.evaluate \
  --csv-file benchmarks/my_benchmark.csv \
  --no-opik \
  --output results/results.json
```

## Programmatic Usage

You can also use the benchmark evaluation in your Python code:

```python
import asyncio
from pathlib import Path
from chatlas_agents.benchmark import run_benchmark

async def main():
    results = await run_benchmark(
        csv_file=Path("benchmarks/my_benchmark.csv"),
        agent_config_file=Path("configs/my-config.yaml"),
        judge_model_name="gpt-5-mini",
        judge_provider="openai",
        output_file=Path("results/results.json"),
        max_items=None,  # Evaluate all items
        use_opik=True,
    )
    
    # Process results
    for result in results:
        print(f"Prompt: {result.prompt}")
        print(f"Average Score: {result.average_score:.2f}/5.0")
        for score in result.scores:
            print(f"  {score.category}: {score.score}/5 - {score.reasoning}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Evaluator

For more control, use the `BenchmarkEvaluator` class:

```python
import asyncio
from pathlib import Path
from chatlas_agents.benchmark import BenchmarkEvaluator
from chatlas_agents.config import load_config_from_env
from chatlas_agents.llm import create_llm_from_config
from deepagents import create_deep_agent

async def main():
    # Load configuration
    config = load_config_from_env()
    
    # Create agent
    agent_llm = create_llm_from_config(config.llm)
    agent = create_deep_agent(model=agent_llm)
    
    # Create judge
    from chatlas_agents.config import LLMConfig, LLMProvider
    judge_config = LLMConfig(provider=LLMProvider.openai, model="gpt-5-mini")
    judge_llm = create_llm_from_config(judge_config)
    
    # Create evaluator
    evaluator = BenchmarkEvaluator(
        agent=agent,
        judge_model=judge_llm,
        agent_config=config.model_dump(),
        use_opik=True,
    )
    
    # Run evaluation
    results = await evaluator.evaluate_from_csv(
        Path("benchmarks/my_benchmark.csv"),
        max_items=10
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Save results
    evaluator.save_results(results, Path("results/results.json"))

if __name__ == "__main__":
    asyncio.run(main())
```

## Output Format

The evaluation results are saved in JSON format:

```json
{
  "timestamp": "2024-12-23T16:30:00.000Z",
  "agent_config": {
    "llm": {
      "provider": "openai",
      "model": "gpt-5-mini"
    },
    "mcp": {
      "url": "https://chatlas-mcp.app.cern.ch/mcp"
    }
  },
  "num_items": 10,
  "average_score": 4.2,
  "results": [
    {
      "prompt": "What is the ATLAS experiment?",
      "expected_answer": "The ATLAS experiment is...",
      "agent_response": "ATLAS (A Toroidal LHC ApparatuS) is...",
      "scores": [
        {
          "category": "accuracy",
          "score": 5,
          "reasoning": "Response is factually correct and comprehensive"
        },
        {
          "category": "relevance",
          "score": 5,
          "reasoning": "Directly answers the question about ATLAS"
        },
        {
          "category": "coverage",
          "score": 4,
          "reasoning": "Covers main points but could mention more details"
        },
        {
          "category": "conciseness",
          "score": 4,
          "reasoning": "Clear and concise with appropriate detail"
        }
      ],
      "average_score": 4.5,
      "metadata": {
        "category": "physics_general",
        "difficulty": "easy",
        "csv_row": 0
      },
      "timestamp": "2024-12-23T16:30:15.000Z"
    }
  ]
}
```

## Opik Best Practices

Following [Opik's agent evaluation best practices](https://www.comet.com/docs/opik/evaluation/evaluate_agents):

### 1. Reproducibility
- All model configurations are logged (model name, provider, temperature, etc.)
- Benchmark CSV files should be version controlled
- Results include timestamps and configuration snapshots

### 2. Comprehensive Logging
- Each agent interaction is tracked with input, output, and metadata
- LLM judge evaluations are logged with scores and reasoning
- Failed evaluations are logged with error details

### 3. Multi-Dimensional Scoring
- Four evaluation dimensions provide nuanced assessment
- Each score includes reasoning for transparency
- Aggregate scores enable trend analysis

### 4. Iterative Improvement
- Compare results across different agent configurations
- Track performance over time as models/prompts change
- Use Opik dashboard to visualize trends

## Command-Line Options

```
usage: evaluate.py [-h] --csv-file CSV_FILE [--config CONFIG]
                   [--judge-model JUDGE_MODEL]
                   [--judge-provider {openai,anthropic,groq}]
                   [--output OUTPUT] [--max-items MAX_ITEMS]
                   [--no-opik] [--verbose]

Run benchmark evaluation on ChATLAS agent

options:
  -h, --help            show this help message and exit
  --csv-file CSV_FILE   Path to CSV file with benchmark data (columns: prompt, expected_answer)
  --config CONFIG       Path to agent configuration YAML file
  --judge-model JUDGE_MODEL
                        Model name for LLM judge (default: gpt-5-mini)
  --judge-provider {openai,anthropic,groq}
                        LLM provider for judge (default: openai)
  --output OUTPUT       Path to save results JSON file
  --max-items MAX_ITEMS
                        Maximum number of items to evaluate
  --no-opik             Disable Opik tracking
  --verbose             Enable verbose logging
```

## CSV Format Requirements

Your benchmark CSV file must have these columns:

- **Required:**
  - `prompt`: The input question/prompt
  - `expected_answer`: The ground-truth answer

- **Optional (metadata):**
  - `category`: Classification of the question type
  - `difficulty`: Difficulty level (easy/medium/hard)
  - Any other columns for filtering/analysis

Example:
```csv
prompt,expected_answer,category,difficulty
"Question 1?","Answer 1",physics,easy
"Question 2?","Answer 2",technical,hard
```

## Troubleshooting

### Opik Not Available

If you see "Opik not available" warnings:
```bash
pip install opik
```

Configure Opik:
```bash
export OPIK_API_KEY=your-api-key
export OPIK_WORKSPACE=your-workspace
```

Or run with `--no-opik` to disable Opik integration.

### Judge Model Errors

If the judge fails to parse scores:
- Check that your judge model supports JSON output
- Increase temperature if responses are too deterministic
- Try a different judge model (e.g., `gpt-5-mini` or `claude-3-5-sonnet-20241022`)

### Agent Timeout

For long-running agent tasks:
- Increase `CHATLAS_MCP_TIMEOUT` environment variable
- Use `--max-items` to test on a smaller subset first
- Check agent logs with `--verbose` flag

## Examples

See the `benchmarks/` directory for example CSV files:
- `example_benchmark.csv` - ATLAS physics and technical questions

## Contributing

To add new evaluation metrics or improve the framework:

1. Extend the `LLMJudge` class for custom scoring
2. Add new columns to CSV for additional metadata
3. Customize the `BenchmarkEvaluator` for specialized workflows
4. Update Opik logging to track additional metrics

## References

- [Opik Documentation](https://www.comet.com/docs/opik/)
- [Opik Agent Evaluation Guide](https://www.comet.com/docs/opik/evaluation/evaluate_agents)
- [DeepAgents Documentation](https://docs.langchain.com/oss/python/deepagents/overview)
