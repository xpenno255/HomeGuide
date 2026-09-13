# Reviewed procedure answers

HomeGuide 0.3.0 includes a reviewed automatic mince-defrost procedure for the
Panasonic NN-ST46KB, UK edition. It uses Chaos Defrost programme 7, entered food
weight and Start. The supported range is 200–1200 g; standing time is separate
from automatic defrost duration.

This record is deliberately narrow. It is not automatic extraction of every
manual's controls. It applies only when:

- An active, successfully indexed manual is confirmed against an appliance with
  manufacturer Panasonic, model NN-ST46KB and region UK.
- The uploaded original file matches the reviewed edition's SHA-256 fingerprint
  in `app/procedures/panasonic_nn_st46kb_uk.json`.
- The question asks about defrosting mince, rather than cooking, manual power/time,
  another food, or a different model.

The source is pages 32–33 of the reviewed UK NN-ST45/46/48 manual. The PDF itself
is not distributed in this repository. A different PDF edition requires another
source review; changing a title or model alone cannot activate this record.
Deleting, disabling or unconfirming the guide removes eligibility immediately.
Reindexing is not needed for this search-time feature.

`/query` returns compact procedure facts, source references, an `answer` and an
answer contract. Existing excerpt clients still receive `results`. Weights are
parsed as one numeric quantity in grams or kilograms, including decimals such as
0.5 kg. Missing, ambiguous or unrecognised quantities request one explicit weight;
out-of-range weights receive an explanation without invented settings.
`POST /api/resolve` accepts `{"question":"microwave defrost 500g mince"}` and returns
only a reviewed answer or `{"status":"not_applicable"}`. It never calls an LLM.
The web `/api/ask` path also uses reviewed answers directly; empty generated
answers for other questions receive an explanatory fallback.

## Reliable spoken answers in Assist

A tool result alone cannot guarantee that a third-party conversation model
speaks it. To remove the measured empty-answer failure for reviewed procedures:

1. In HomeGuide's integration options, open **Assist conversation agent** and
   select your existing conversation entity (for example, your Gemma agent).
2. Keep HomeGuide's LLM tool enabled on that existing agent for general manual
   questions.
3. In **Settings → Voice assistants**, select **HomeGuide Assist** as the
   conversation agent for the assistant you want to use.

The HomeGuide agent speaks verified procedure answers directly. Other requests
are forwarded to the selected agent with their text, context and conversation
ID preserved. There is no modification to Extended OpenAI Conversation or vLLM.
The wrapper is optional; existing tool-only installations keep working.

Reviewed answers currently support English and self-contained questions. Include
the appliance and weight each time ("microwave defrost 750g mince"); pronoun-only
follow-ups such as "what about 750g?" are delegated to the existing agent. Direct
reviewed answers are not inserted into that agent's private chat history. A
HomeGuide agent cannot be selected as its own delegate.

Update the **backend container as well as the HA integration** to 0.3.0. HACS
manages only the HA integration; it does not install Docker or your manuals.
