---
# The tables are wide, so the page leaves out the navigation on its left and
# the table of contents on its right.
hide:
- navigation
- toc
---

# Pipeline Chart

Every named pipeline in `amos`, step by step. A pipeline fills in the
settings of a project's three workers: the **analyst** prepares the data and
fits the models, the **critic** cross-validates and evaluates them, and the
**artist** draws the figures. Name one with `pipeline` in the "general"
section (see [pipeline](idea.md#pipeline)), and every setting that you write
yourself wins over the pipeline's:

```ini
[general]
label = target
seed = 43
pipeline = classification
```

A step whose technique is `auto` adapts to the data when the project runs
(see the choosers in the [technique catalog](catalog.md#choosers-any-stage)).
The catalog's [pipelines](catalog.md#pipelines) table describes each
pipeline in a sentence.

```python exec="on"
import runpy

print(runpy.run_path('docs/scripts/pipeline_chart.py')['chart']())
```
