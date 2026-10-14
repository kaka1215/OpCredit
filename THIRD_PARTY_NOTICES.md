# Third-party notices

`src/opcredit/training/actor.py` adapts the update loop in
`verl/workers/actor/dp_actor.py` from veRL 0.7.1
(https://github.com/verl-project/verl), licensed under Apache-2.0.

Copyright 2024 Bytedance Ltd. and/or its affiliates.
Copyright 2023-2024 SGLang Team.
Copyright 2025 ModelBest Inc. and/or its affiliates.

The adaptation adds the OpCredit joint objective, fixed rollout credits, and
sequence-mean micro-batch accumulation. The original license is included in
`licenses/verl-Apache-2.0.txt`. Other dependencies are installed separately and
retain their respective licenses. Benchmark data and model weights are not
redistributed in this package.
