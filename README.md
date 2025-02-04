# PCL-lite
Light-weight STeP in Python

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
2. Supported models:

| Model | API |
|-------|-----|
| DeepSeek-V3 | DeepSeek-chat |
| gpt-4o-2024-11-20 | OpenAI |
| claude-3-5-sonnet-20241022 | Anthropic |
| Meta-Llama-3-1-405B-Instruct-Turbo | TogetherAI |
| Qwen2-5-Coder-32B-Instruct | TogetherAI |
