# Context budget core

This package owns two provider-call boundaries:

1. `IncrementalContextBudgetEstimator` measures content blocks by UTF-8 hash and
   reuses unchanged token counts across current-valid projection rebuilds;
   `IncrementalContextBlockBudget` then admits blocks in caller-supplied policy
   order, rejects optional overflow without spending budget, and fails closed
   when a mandatory block cannot fit.
2. A trusted, physical-route-bound `ProviderEnvelopeTokenMeter` measures the
   exact prepared wire bytes. `admit_context_request` invokes that meter itself,
   checks output/reasoning reserves, configured input limit, provider context
   window, body bytes, route, and component charges, then returns an ephemeral
   `AdmittedContextRequest` containing the only bytes that may be dispatched.

It deliberately does not retrieve documents, rank evidence, call embedding or
generation models, choose a provider route, mutate a projection, or perform
provider I/O. Those operations remain with their owning adapters. Receipts and
block measurements contain hashes and bills, never prompt content.

`purpose`, `projection_epoch`, and `projection_generation` are caller-supplied
audit provenance, not provider-derived facts. They are bound into the admitted
receipt's authority record and cannot be relabelled after the gate issues a
dispatch request.

`configured_input_limit_tokens` is an input-only product cap. It is not a second
full context window and therefore does not subtract output reserves twice. The
effective hard input budget is:

```text
min(
  configured_input_limit_tokens,
  provider_context_window - output_reserve - reasoning_reserve - safety_margin,
)
```

Provider adapters may use `measure_envelope_with_meter` for diagnostics after
applying their exact chat-template and multimodal rules.
`measure_canonical_json_envelope` is a conservative semantic-JSON fallback, not
proof of the bytes emitted by an HTTP library. The lower-level
`measure_precounted_envelope*` helpers are estimate/adapter test seams and cannot
manufacture dispatch authority or exactness; provider-derived component labels
require a provider-derived meter identity.

For a live integration, prepare the complete wire payload first (including a
streaming flag), serialize it once, call `admit_context_request` with the
route-bound provider meter, and dispatch only
`AdmittedContextRequest.body_for_dispatch()`. The gate meters those same bytes;
pre-counted or canonical fallback measurements cannot create an admitted
request. The gate must run before durable logical or physical attempt
reservation.

Inline base64 images remain part of the exact admitted bytes and body-size
limit, but are not counted as ordinary prompt text. The provider adapter
validates the dialect-specific image shape, replaces only the meter copy's raw
base64 with a sentinel, verifies PNG/JPEG bytes and safe dimensions, then adds
one separately labelled heuristic vision-token component per image using the
larger of byte- and pixel-grid estimates. A route-certified pixel/token meter
can later replace that heuristic without changing the core admission contract.

When the configured text counter has fallen back to the legacy heuristic, the
final wire gate raises its text bill to at least one token per UTF-8 byte. That
is intentionally more pessimistic than the first-level projection estimate: it
prevents emoji, unfamiliar scripts, and high-entropy text from exploiting a
char/4 underestimate. A real configured tokenizer retains tokenizer provenance;
neither path is labelled provider-exact.

Within this package, `token_counter.py` owns environment-driven block caps and
text truncation. The remaining modules own the per-call measurement/admission
contracts. Both assembly and provider paths share the single
`ContextBudgetExceeded` error type.

## Current production boundary

The repository's built-in Anthropic-compatible and OpenAI-compatible chat
factories (streaming and non-streaming) use this final gate and reuse its exact
admitted bytes for HTTP dispatch. This is the supported production path.

The following are explicit hardening work, not properties this version claims:

- Runtime provider injection remains a compatibility/test seam. A plain
  callable without `.prepare` follows the legacy path, and a third-party
  prepared object is structurally checked rather than cryptographically tied to
  `AdmittedContextRequest`. Do not use either extension seam as a production
  provider boundary until it requires a sealed admission proof.
- A durable successful replay is currently prepared and re-admitted before its
  stored typed result is discovered; a profile/configuration change can
  therefore block a replay that would otherwise perform no Provider I/O.
  Restarted output repair also re-admits the original variant before recovering
  the durable repair variant.
- Context windows, vision support, and vision charges come from conservative
  local model-family profiles. Custom aliases and new provider models require a
  route-certified profile/token meter before the result can be treated as an
  actual-provider upper bound. The current meter intentionally never labels
  these estimates provider-exact.
- The durable physical ledger does not yet bind the admitted wire SHA-256, so
  cross-version recovery proves logical/state identity but not byte-for-byte
  provider-request identity.
