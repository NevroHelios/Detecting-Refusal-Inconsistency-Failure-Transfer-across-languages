"""Batched scoring must equal the sequential per-prefix reference.

Uses fake tokenizer/model so it runs on CPU without downloading weights. The
fake model's logits are a deterministic function of (token_id, position) only,
so right-padding a batch leaves each real token's logits untouched -- exactly
the invariant a causal model + attention_mask provides. That lets us assert
batched == sequential and catch any off-by-one in the gather indices."""

import torch

from src import score


class FakeTok:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, msgs, add_generation_prompt=True,
                            return_dict=False, **kw):
        text = "|".join(m["content"] for m in msgs)
        return [1] + [(ord(c) % 50) + 3 for c in text]  # BOS=1, never 0 (pad)

    def __call__(self, text, add_special_tokens=False):
        class R:
            pass
        r = R()
        r.input_ids = [(ord(c) % 50) + 3 for c in text]
        return r


class FakeModel:
    device = "cpu"
    vocab = 64

    def __call__(self, ids, attention_mask=None):
        B, L = ids.shape
        t = torch.arange(L).view(1, L, 1).float()
        tok = ids.view(B, L, 1).float()
        v = torch.arange(self.vocab).view(1, 1, self.vocab).float()

        class O:
            pass
        o = O()
        o.logits = torch.sin(tok * 0.7 + v * 0.13 + t * 0.05)
        return o


def test_score_prompt_matches_sequential_reference():
    tok, model = FakeTok(), FakeModel()
    prompt = "how do I do the thing"
    comply = ["Sure here", "Okay I can help"]
    refuse = ["Sorry no", "I cannot", "Not able"]

    # Ground truth: the existing single-prefix path, run once per prefix.
    ref_c = [score.prefix_logprob(tok, model, prompt, p)[0] for p in comply]
    ref_r = [score.prefix_logprob(tok, model, prompt, p)[0] for p in refuse]
    exp_comply = sum(ref_c) / len(ref_c)
    exp_refuse = sum(ref_r) / len(ref_r)

    out = score.score_prompt(tok, model, prompt, comply, refuse)

    assert out["lp_comply"] == torch.tensor(exp_comply).item() or \
        abs(out["lp_comply"] - exp_comply) < 1e-5
    assert abs(out["lp_refuse"] - exp_refuse) < 1e-5
    assert abs(out["s"] - (exp_comply - exp_refuse)) < 1e-5
    assert out["n_prompt_tokens"] == score.prefix_logprob(tok, model, prompt, comply[0])[1]


def test_batch_prefix_logprob_one_forward_pass():
    """All prefixes scored in a single model call, not one call per prefix."""
    tok = FakeTok()
    calls = {"n": 0}

    class Counting(FakeModel):
        def __call__(self, ids, attention_mask=None):
            calls["n"] += 1
            return super().__call__(ids, attention_mask)

    model = Counting()
    score.batch_prefix_logprob(tok, model, "prompt", ["a", "bc", "def"])
    assert calls["n"] == 1
