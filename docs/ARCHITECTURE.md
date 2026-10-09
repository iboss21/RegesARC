# Architecture

One process. Two placement paths. No nested repos.

```
reges serve
    scan  ->  gpu, ram, disk
    pick  ->  catalog verdict
    place ->  card  |  disk
    serve ->  OpenAI + Anthropic on one port
```

Card path. Dense trunk and hot experts live on the GPU. Cold experts stream.
This is the Strata mechanism, and it is how a 125B MoE moves on 12 GB.

Disk path. Dense trunk stays in RAM. Routed experts stay on the SSD and load
only when the router selects them. This is the Colibri mechanism, and it is
how a 170B or 744B MoE answers when the card cannot hold the hot set.

The scanner picks the path. The user does not.

```
card can hold the hot set     -> card path
card cannot, RAM holds trunk  -> disk path
neither                       -> refuse, do not pretend it fits
```

Engines land in `engines/` as source absorbed from a pinned upstream tag,
recorded in NOTICE. Weights land in `RegesModels/` beside the checkout.
Neither is a git submodule.
