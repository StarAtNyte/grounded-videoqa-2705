# Grounded VideoQA

Language-guided object grounding and tracking for the Perception Test 2026 Grounded VideoQA track.

Video reasoning → answer-object grounding → bidirectional tracking → normalized bounding-box tracks and validation.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --extra dev
uv run pytest
uv run modal run -m grounded_videoqa.jobs.modal_app --help
```

The local test suite uses synthetic inputs. Model inference runs in Modal;
GPU frameworks and model-specific dependencies are defined in each job's image,
so local setup does not download model weights.

## Running the pipeline

Authenticate with Modal and configure the profile and any model/API secrets required
by the selected job. The commands below launch remote work and may incur cloud costs.

```bash
uv run modal run -m grounded_videoqa.jobs.modal_app::download --split valid
uv run modal run -m grounded_videoqa.jobs.modal_app::inspect --split valid
```

Use module paths for every job: `uv run modal run -m grounded_videoqa.jobs.<module>`.
Local tools also run as modules:

```bash
uv run python -m grounded_videoqa.tools.validate /path/to/predictions.json
```

## Code layout

| Path | Purpose |
| --- | --- |
| [`src/grounded_videoqa/`](src/grounded_videoqa/) | Shared components: `gqa_core.py`, reasoning, grounding models, tracker backends, and `track_graph.py` |
| [`src/grounded_videoqa/jobs/`](src/grounded_videoqa/jobs/) | Modal data jobs, inference pipelines, and experiments |
| [`src/grounded_videoqa/tools/`](src/grounded_videoqa/tools/) | Local evaluation and operational utilities |
| [`tests/`](tests/) | Synthetic regression tests |
| [`docs/`](docs/README.md) | Research, architecture, experiment history, and recorded results |

The previous root-level Python modules now live under `src/grounded_videoqa/`.
Imports use the package namespace, and Modal images include that package for remote execution.

## Data and evaluation

Persistent storage: `grounded-videoqa-2705-data`, mounted at `/data`. Keep raw media, extracted frames, model weights,
credentials, and new prediction artifacts out of Git. Existing checked-in experiment
records are retained for traceability.

Read [`AGENTS.md`](AGENTS.md) for challenge-specific constraints. Preserve organizer
IDs, option order or track identities, and the submission schema. Validation and test
examples must not become training or few-shot prompt data. Grounded outputs contain
box tracks; multiple-choice outputs contain the selected option.

## Development and project records

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for verification and contribution guidance.
Build a distributable package with `uv build`. CI runs the synthetic test suite and
package build without cloud credentials.

Research notes and score tables in [`docs/`](docs/README.md) are dated experiment
records; they do not represent a freshly verified leaderboard. See the
[official challenge](https://eval.ai/web/challenges/challenge-page/2705/overview)
for evaluation details.
