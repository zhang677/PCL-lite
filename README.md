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

## Pitfalls
1. We recommend changing the `BASE_PATH` in the `experiments` bash scripts to folder that are not git. Otherwise, parallel sampling might be slowed down by more than 10x because of git logging.