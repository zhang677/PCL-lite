# PCL-lite
An adaptive self-improvement LLM agentic system for ML library development. We choose [STeP](https://ppl.stanford.edu/papers/YARCH24_STEP.pdf) as the target ASPL for next-generation RDA.

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

## Iterative experimnent
```
./experiments/iterative/run.sh
```

## Guide
1. We recommend changing the `BASE_PATH` in the `experiments` bash scripts to folder that are not git. Otherwise, parallel sampling might be slowed down by more than 10x because of git logging.
2. Users can change the `MODEL_NAME` in the `experiments` bash scripts to any supported model:

| Model | API |
|-------|-----|
| claude-3-5-sonnet-20241022 | Anthropic |
| gpt-4o-2024-11-20 | OpenAI |
| Meta-Llama-3-1-405B-Instruct-Turbo | TogetherAI |
| DeepSeek-V3 | DeepSeek-chat |
| Qwen2-5-Coder-32B-Instruct | TogetherAI |
3. Since STeP is still a research prototype, we only public the bmm tasks in the benchmark.

## Cite
