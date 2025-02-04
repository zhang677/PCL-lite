# PCL-lite
An adaptive self-improvement LLM agentic system for ML library development. We choose [STeP](https://ppl.stanford.edu/papers/YARCH24_STEP.pdf) as the target ASPL for next-generation RDA. Please run the following commands in order and under the `/PCL-lite` folder.

(Optional)
```
pip install -r requirements.txt
```

## Validation

Validate the test and reference yamls under `/benchmark` and `/prompts`
```
./scripts/validate.sh
```

## Prepare benchcard
```
./scripts/prepare.sh
```

## Single experiment
```
./experiments/single/run.sh
```

## Agent experiment
```
./experiments/agent/run.sh
```

## Self-improvement agent experimnent
```
./experiments/iterative/run.sh
```

## Single without structual IR experiment
```
./experiments/single-ws/run.sh
```

## Guide
1. We recommend changing the `BASE_PATH` in the `experiments` bash scripts to folder that are not git. Otherwise, parallel sampling might be slowed down by more than 10x because of git logging.
2. Users can change the `MODEL_NAME` in the `experiments` bash scripts to any supported model:

| Model | API | Environment Variable |
|-------|-----|-----|
| claude-3-5-sonnet-20241022 | Anthropic | ANTHROPIC_API_BASE, ANTHROPIC_API_KEY |
| gpt-4o-2024-11-20 | OpenAI | OPENAI_API_BASE, OPENAI_API_KEY |
| Meta-Llama-3-1-405B-Instruct-Turbo | TogetherAI | TOGETHER_API_BASE, TOGETHER_API_KEY |
| DeepSeek-V3 | DeepSeek-chat | DEEPSEEK_API_BASE, DEEPSEEK_API_KEY |
| Qwen2-5-Coder-32B-Instruct | TogetherAI | TOGETHER_API_BASE, TOGETHER_API_KEY |
3. Since STeP is still a research prototype, we only publish the bmm tasks in the benchmark.
4. `NUM_SAMPLES` and `TEMPERATURE` can be adjusted.

## Cite
