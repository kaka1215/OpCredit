# OpCredit

Implementation of operation-aware policy learning for agentic KGQA, including
KG-grounded data preparation, template-based SFT, and on-policy RL. SFT uses
Transformers; RL uses veRL with FSDP workers and async vLLM rollouts. The adapter
extends veRL without editing its installed files.

## Contents

| Component | Role |
| --- | --- |
| `data` | Dataset adapters, path grounding, and SFT/RL data preparation |
| `agent` | KG interaction, prompts, model interface, and rollout collection |
| `credit` | BOC, FTC, decision-to-token alignment, and joint objective |
| `training` | SFT/RL entry points, terminal task reward, and veRL update adapters |
| `records` | Shared state, operation, decision, and rollout records |
| `configs` | Unfilled configuration templates |
| `scripts` | Data preparation, SFT, RL, and ablation launchers |

## Method implementation

| Paper component | Implementation |
| --- | --- |
| Terminal task reward | `training/reward.py` |
| Navigation context and decision spans | `records.py` |
| Branch Outcome Credit | `credit/boc.py` |
| Frontier Transition Credit | `credit/ftc.py` |
| Alignment to generated tokens | `credit/alignment.py` |
| Joint objective and dual clipping | `credit/objective.py` |
| Rollout collection | `agent/rollout.py` |
| Group credit assignment and policy update | `training/advantages.py`, `training/actor.py` |

During RL, each rollout returns its decision records and training-only answer
neighborhood. The trainer groups the records for BOC and uses each rollout's
own neighborhood for FTC. Both credits are aligned to the same generated-token
mask before the veRL actor applies the joint objective.

## Installation and configuration

The experiment stack used PyTorch 2.11.0, Transformers 5.5.4, veRL 0.7.1,
and vLLM 0.21.0. RL requires a separately provisioned veRL environment.

Install from the package root in the configured environment:

```bash
python -m pip install -e '.[train,rl]'
```

The `rl` extra installs the declared vLLM stack, not veRL. For data preparation or
SFT without RL, install `.[train]`. Install the `distributed` extra if using
DeepSpeed for SFT and supply its configuration through `--deepspeed`.

The JSON templates contain field names with `null` values. They do not include
experiment settings, model locations, dataset locations, or service addresses.
Copy the appropriate template to a private configuration file and fill the
required fields before running. Missing required values cause an argument error.
Optional `resume` and `deepspeed` fields can remain `null`. The `cpu` switch is
optional; both BOC and FTC are enabled unless explicitly disabled.

Use `--help` on an entry point to inspect its required and optional arguments.
JSON field names use underscores; command-line options use hyphens. Command-line
values override JSON values. For repeated data sources, use lists of
`dataset=location` strings in the `input` and `exclude` fields. No runtime
configuration, generated data, or checkpoints are distributed in this package.

## Launchers

Run from the package root after installation. Set `CONFIG_FILE` to your private
configuration file for the relevant stage:

```bash
bash scripts/prepare_data.sh --config "$CONFIG_FILE"
bash scripts/train_sft.sh --config "$CONFIG_FILE"
bash scripts/train_rl.sh --config "$CONFIG_FILE"
```

The scripts forward arguments without preset experiment values. `PYTHON_BIN`
can select the Python executable. For distributed SFT on one node, supply
`NPROC_PER_NODE` in the environment; otherwise the SFT launcher runs directly.
RL starts Ray workers through veRL and should not be wrapped in `torchrun`.
The KG service and supplied resources must be accessible to those workers.

Use the ablation launcher with a separately supplied configuration and output
location for each run:

```bash
bash scripts/run_ablation.sh "$VARIANT" --config "$CONFIG_FILE"
```

| Variant | Outcome objective | FTC objective |
| --- | --- | --- |
| `grpo` | GRPO | Disabled |
| `boc` | BOC | Disabled |
| `ftc` | GRPO | Enabled |
| `opcredit` | BOC | Enabled |

The selected variant takes precedence over the BOC/FTC switches in the
configuration. Other settings are forwarded unchanged.

## Data preparation and SFT

Provide a Freebase SPARQL endpoint and a SentenceTransformer-compatible embedding
model. The loader accepts native WebQSP and GrailQA records, and preprocessed
CWQ records containing a question identifier, question text, topic entity, answer
entities, and gold relation sequences. Optional oracle paths supply alternative
relation sequences and similarity scores. Retain CWQ's `webqsp_ID` to identify its
source question. Supply evaluation records for overlap exclusion; they are not
used as training examples. Prepare GrailQA separately from the CWQ/WebQSP pool.

Gold paths are grounded against the supplied KG before constructing multi-turn
SFT traces. Reasoning text is generated from templates without a teacher model.
The KG interface expands unnamed intermediate nodes by one additional edge in
both directions and returns named endpoints. This remains one interface navigation
hop; FTC proximity uses the separately constructed one-edge KG neighborhood.
Data preparation and online navigation use the same expansion rule. The system
and observation prompts follow the templates in the paper's appendix.

Supervised choices can be retained in retrieved candidate lists during data
construction. Online RL retrieval does not receive gold paths or answer labels.
Incomplete questions and ungroundable paths are counted in the preparation
report. Use the same prepared data across policy backbones.

SFT supervises assistant reasoning and operation tokens. Questions, observations,
and padding are masked. Over-length trajectories are counted and excluded as
complete traces. The effective batch size is distributed across processes and
gradient accumulation. All training settings and output locations are supplied
at runtime. Resume a training checkpoint through `--resume`.

## Reinforcement learning

RL collects complete same-question rollout groups and assigns both signals to
each generated decision span, including its reasoning and final answer emission.
BOC restricts both Q and V to the same question and active entity, then matches
visited entities, remaining budget, and decision phase. Each trajectory is counted
once per support set. Nonzero context-relative outcome contrasts provide local
credit; zero contrasts retain the corresponding trajectory advantage.

FTC compares each operation's before/after frontier using the answer-conditioned
potential. Answer emission uses the full predicted answer entity set and its
maximum potential. The bounded answer neighborhood includes both edge directions;
retrieval limits are supplied at runtime. Rewards and FTC are computed after
rollout collection. Unmapped frontiers receive no FTC signal.

The outcome and FTC objectives first average generated tokens within each
trajectory, then average trajectories. Micro-batch accumulation uses the same
weighting. Observations are excluded from the policy loss. The frozen reference
policy is the supplied SFT initialization. Checkpoints use veRL's training-state
format; resumption requires the original SFT initialization and a supplied
checkpoint location.

Full training requires compatible GPUs, model weights, a KG, and benchmark
resources. Evaluation scripts and internal validation materials are not bundled.
Third-party notices and license texts are retained.

## License

Copyright 2026 OpCredit contributors. Licensed under [Apache-2.0](LICENSE).
Third-party attributions are listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
