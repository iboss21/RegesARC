# Engine contract

An engine is one family, one binary, three commands.

```
reges-engine info   <model-dir>
reges-engine serve  <model-dir> --port <n> --place card|disk
reges-engine chat   <model-dir> --place card|disk
```

`serve` speaks OpenAI `/v1/chat/completions` and Anthropic `/v1/messages`.
Placement is chosen by the launcher, not by the engine's own menu.

Nothing in this directory is upstream source yet. When a family is absorbed,
the commit names the upstream tag and NOTICE is updated in the same commit.
