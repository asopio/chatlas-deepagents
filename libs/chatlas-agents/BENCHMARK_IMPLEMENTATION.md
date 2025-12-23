# Benchmark Evaluation Implementation - Opik Best Practices

## Summary

This implementation provides a comprehensive benchmark evaluation framework for ChATLAS deep agents, following Opik best practices for agent evaluation and reproducibility.

## Key Components

### 1. Opik-Native Metrics

Instead of a custom LLM-as-judge implementation, we use Opik's built-in, well-tested metrics:

- **AnswerRelevance**: Evaluates if the agent's response is relevant to the input prompt
- **Hallucination**: Detects factual errors and inconsistencies (mapped to "accuracy")
- **GEval** (Custom): Two custom metrics for coverage and conciseness using Opik's GEval framework

### 2. Four Evaluation Dimensions

The framework evaluates agent responses across four key dimensions:

1. **Accuracy** (via Hallucination metric): Is the response factually correct?
2. **Relevance** (via AnswerRelevance metric): Does it address the user's question?
3. **Coverage** (via custom GEval): Does it cover all important aspects?
4. **Conciseness** (via custom GEval): Is it clear and concise?

### 3. Opik Integration Features

#### Experiment Tracking
- Uses `@track` decorator for automatic tracing of agent execution
- Stores agent configuration in experiment metadata
- Groups related runs using experiment names
- Organizes benchmarks using project names

#### Reproducibility
- All model configurations logged (model name, parameters, etc.)
- Benchmark CSV files serve as versioned datasets
- Results include timestamps and configuration snapshots
- Opik dashboard provides visualization and comparison tools

#### Score Format
- Follows Opik standard: 0.0-1.0 float scores (not 1-5 integers)
- Each score includes:
  - `value`: Float score between 0.0 and 1.0
  - `reason`: Detailed explanation from the judge LLM
  - `name`: Metric identifier

### 4. Workflow

```
1. Load benchmark CSV → Prompts + Expected Answers
2. For each item:
   a. Run agent with @track decorator
   b. Evaluate response with Opik metrics
   c. Log to Opik dashboard
3. Aggregate results and display summary
4. Save JSON results locally
5. View detailed analysis in Opik dashboard
```

### 5. CSV Format

```csv
prompt,expected_answer,category,difficulty
"What is ATLAS?","ATLAS is a particle physics detector...",physics,easy
"How to use Rucio?","Rucio is the data management system...",technical,medium
```

Required columns:
- `prompt`: Input question
- `expected_answer`: Ground-truth answer

Optional metadata columns (e.g., `category`, `difficulty`) are preserved in results.

## Alignment with Opik Best Practices

### From Opik Documentation Research

1. **Use Built-in Metrics** ✅
   - Leverages Opik's pre-built AnswerRelevance and Hallucination metrics
   - Uses GEval for custom evaluation dimensions

2. **Proper Experiment Structure** ✅
   - Experiments have names and metadata
   - Projects organize related experiments
   - Configuration stored for reproducibility

3. **Tracing & Logging** ✅
   - `@track` decorator on agent execution
   - `configure()` for Opik initialization
   - All evaluations logged to dashboard

4. **Multi-Dimensional Scoring** ✅
   - Four evaluation dimensions for nuanced assessment
   - Each dimension has reasoning for transparency
   - Aggregate scores enable trend analysis

5. **Iterative Improvement** ✅
   - Compare results across agent configurations
   - Track performance over time
   - Dashboard visualization for trends

## Usage Examples

### Basic Benchmark
```bash
chatlas benchmark \\
  --csv-file benchmarks/example_benchmark.csv \\
  --output results.json
```

### Full Featured
```bash
chatlas benchmark \\
  --csv-file benchmarks/atlas_qa_v2.csv \\
  --config configs/claude-agent.yaml \\
  --judge-model claude-3-5-sonnet-20241022 \\
  --experiment-name "atlas-qa-v2-eval" \\
  --project-name "chatlas-production" \\
  --output results/2024-12-23-eval.json \\
  --verbose
```

### Python API
```python
from chatlas_agents.benchmark import run_benchmark
from pathlib import Path

results = await run_benchmark(
    csv_file=Path("benchmarks/my_benchmark.csv"),
    judge_model="gpt-5-mini",
    use_opik=True,
    experiment_name="test-run-001",
    project_name="chatlas-dev",
)
```

## Output Format

### JSON Results
```json
{
  "timestamp": "2024-12-23T16:30:00.000Z",
  "agent_config": {
    "llm": {"provider": "openai", "model": "gpt-5-mini"},
    "mcp": {"url": "https://chatlas-mcp.app.cern.ch/mcp"}
  },
  "num_items": 10,
  "average_score": 0.85,
  "results": [
    {
      "prompt": "What is ATLAS?",
      "expected_answer": "ATLAS is a particle detector...",
      "agent_response": "ATLAS (A Toroidal LHC ApparatuS) is...",
      "scores": {
        "relevance": {
          "value": 0.95,
          "reason": "Response directly answers the question..."
        },
        "accuracy": {
          "value": 0.90,
          "reason": "Response is factually correct..."
        },
        "coverage": {
          "value": 0.80,
          "reason": "Covers main points but could include..."
        },
        "conciseness": {
          "value": 0.75,
          "reason": "Clear but slightly verbose..."
        }
      },
      "average_score": 0.85,
      "metadata": {"category": "physics", "difficulty": "easy"}
    }
  ]
}
```

### Terminal Output
```
Benchmark Evaluation Summary
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric           ┃  Score ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Overall Average  │  0.850 │
│                  │        │
│ Accuracy         │  0.900 │
│ Conciseness      │  0.750 │
│ Coverage         │  0.800 │
│ Relevance        │  0.950 │
│                  │        │
│ Total Items      │     10 │
└──────────────────┴────────┘

✓ Results saved to results/results.json

View detailed results in Opik dashboard: https://www.comet.com/
```

## Benefits

1. **Reliability**: Uses Opik's tested and validated metrics
2. **Consistency**: Standard 0.0-1.0 scoring across all metrics
3. **Transparency**: Each score includes detailed reasoning
4. **Reproducibility**: Full configuration and data versioning
5. **Visualization**: Opik dashboard for exploring results
6. **Comparison**: Easy comparison across experiments
7. **Monitoring**: Track performance trends over time

## Future Enhancements

Potential improvements aligned with Opik best practices:

1. **Trajectory Evaluation**: Evaluate the agent's tool usage patterns
2. **Online Evaluation**: Real-time evaluation in production
3. **Human Feedback**: Align metrics using human labels (Opik's Align Evaluator)
4. **Custom Datasets**: Create Opik datasets for reusable benchmarks
5. **Pairwise Evaluation**: Compare two agent versions side-by-side
6. **Batch Evaluation**: Evaluate multiple configurations simultaneously

## References

- Opik Documentation: https://www.comet.com/docs/opik/
- Opik Agent Evaluation Guide: https://www.comet.com/docs/opik/evaluation/evaluate_agents
- Opik Metrics Library: https://www.comet.com/docs/opik/evaluation/metrics/
- Example Benchmark CSV: `benchmarks/example_benchmark.csv`
- Detailed Documentation: `chatlas_agents/benchmark/README.md`
