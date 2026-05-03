# commaVQ Task Renderer

This prototype tests a pretrained driving-video token path:

```text
original frames --commaVQ encoder, compression side only--> 8x16 uint10 tokens
tokens --small task renderer in archive--> evaluator-facing frame pair
```

The commaVQ encoder/decoder weights are used only for offline tokenization and oracle evaluation. They are not used by `inflate.py` and should not be included in a submitted archive unless a future candidate explicitly depends on them.

Run order:

```text
1. encode_tokens.py on hard8 or 64 samples
2. eval_commavq_decoder.py as the real-decoder information oracle
3. train_renderer.py only if the real decoder oracle is not dead
4. pack_tokens.py / pack_renderer.py only if 64-sample renderer capacity passes
```

Kill gates:

```text
real commaVQ decoder quality >0.300: stop
hard8 renderer quality >0.300 after 5k steps: stop
64-sample renderer quality >0.180: stop before packing/rate work
```

