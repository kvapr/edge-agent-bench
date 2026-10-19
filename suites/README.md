# Test suites

The suites test whether a model can turn a natural-language request into the
correct structured tool call for an agent that operates a managed PoE switch.

## Files

- `test_cases.yaml` holds the 65 canonical cases.
- `test_cases_paraphrase.yaml` holds 195 paraphrases: each canonical case
  rewritten in three styles.
- `tools_nested.json` is the 12-tool schema in OpenAI function format; runs offer
  a 4-tool subset (below). `tools_flat.json` and `test_cases_flat.yaml` are a
  flat-schema variant used once to check that schema shape is not a factor.

## Canonical cases

The 65 cases fall into four levels of difficulty.

| Level | Cases | What the request asks for |
|---|---|---|
| 1, queries | 20 | Read-only status, for example "@poesw1 show port 5 status" |
| 2, enable or disable | 15 | Switch a port or its PoE power on or off |
| 3, single parameter | 15 | One setting, for example a port speed |
| 4, multiple parameters | 15 | Several values, for example a port and a VLAN ID |

Each case records the expected tool, the expected command and the expected
parameters. Across the suite, 14 distinct commands are used. Twenty cases
expect `query_device_info` and 45 expect `configure_device_setting`.

## Paraphrases

Every canonical case has three paraphrases, 65 per style, each carrying exactly
its base case's expected tool, command and parameters.

- Style A is casual and terse ("@poesw1 what's going on with port 5?").
- Style B is polite and formal ("Would you mind pulling up the current status
  for port 5, @poesw1?").
- Style C is indirect and embeds the request in a situation, so the model has
  to infer the operation ("@poesw1 port 11 is drawing more poe than we'd like,
  restrict it to 20 watts", lightly abridged).

## Request and tools

Each request carries the system prompt in `harness/bench_toolcall.py`, one user
message and four tools: `query_device_info`, `configure_device_setting`,
`diagnose_port_issues` and `search_documentation`. No case expects the last two,
so they act as distractors. Rendered through a model's chat template, one
request is about 2,700 tokens, of which the four tool definitions are about
2,230. The subset is kept to four tools because the compiled NPU bundles have a
4,096-token context; the full 12-tool schema would not fit alongside an answer.

The system prompt states the conventions the labels follow. The device named
after `@` is passed as `device_name` without the `@`; ports use the `1/N` form;
VLAN IDs and PoE power limits are integers. PoE commands are expected only when
the request concerns PoE power; a request to turn a port on or off maps to
`enable_port` or `disable_port`.

## Grading

A case passes end to end when the model returns a structured tool call with the
expected tool, the expected command, the target device, and every expected
parameter present with an equal value (compared after string normalization;
extra parameters are ignored). Tool selection, command choice and strict
parameter equality are also reported on their own. A tool call written as text
in the message body rather than returned as a structured call fails, because
an agent reading the structured field would receive nothing; the harness flags
such replies so the failure mode is visible.

## Validation

`analysis/check_suites.py` checks the suites mechanically. Every expected
command is one the system prompt names; every port is in `1/N` form;
every VLAN ID and power limit is an integer; and every paraphrase carries
exactly the labels of its base case. As of 2026-10-14 all checks pass with no
exceptions.
