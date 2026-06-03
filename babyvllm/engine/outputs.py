"""
Streaming output data structure:
  Encapsulates token sequences produced by the engine layer into routable output objects.

In online inference flow:
  1. LLMEngine.step() produces a list of (seq_id, token_ids)
  2. AsyncLLMEngine encapsulates it into RequestOutput objects
  3. RequestTracker routes RequestOutput to the corresponding request's AsyncStream
  4. API layer pushes chunks to clients via async generator

Relationship with offline mode:
  Offline mode (LLMEngine.generate()) bypasses RequestOutput and directly returns
  results in dict format. The two modes operate independently.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# RequestOutput: Atomic data unit for single step output
#
# After each step that emits generated tokens, the engine produces a
# RequestOutput for each request that made progress. Chunked prefill pieces that
# have not completed the prompt are intentionally silent because their sampled
# token is not valid yet. If finished=True, this is the final output for that
# request; otherwise, subsequent decode steps may continue producing outputs.
#
# Example: prompt "Hello, my name is" generating " Bob, nice to meet you":
#   First emitted step (decode, token=" Bob"):
#     RequestOutput(request_id=1, text=" Bob", token_ids=[9473],
#                   finished=False,
#                   prompt_token_ids=[15496, 11, 1070, 1674, 374])
#   Next decode step (token=","):
#     RequestOutput(request_id=1, text=",", token_ids=[11],
#                   finished=False, ...)
#   ...
#   Final Step (eos):
#     RequestOutput(request_id=1, text=" you", token_ids=[937],
#                   finished=True, ...)
# ---------------------------------------------------------------------------

@dataclass
class RequestOutput:
    """
    Output segment for a single request at a specific step.

    Attributes:
        request_id:
            Unique identifier for the request, from AsyncLLMEngine._request_counter.
            Used by RequestTracker to route output to the correct AsyncStream.
            NOT the same as Sequence.seq_id — the engine maintains its own ID
            space (via itertools.count) independent of the scheduler's sequence
            IDs (from Sequence.counter). The mapping between the two is maintained
            in AsyncLLMEngine._seq_to_request and _request_to_seq.

        text:
            Text corresponding to tokens generated in this step (detokenized).
            Empty for silent bookkeeping outputs; otherwise usually text for
            the generated token delta from a decode step.

        token_ids:
            List of newly generated token IDs in this step.
            Empty list for steps that generate no completion token; typically
            [single_token_id] during decode.
            Note: Does not include prompt token IDs, only newly generated ones.

        finished:
            Whether the request has completed (EOS generated, max_tokens reached, etc.).
            When finished=True, this is the final output for the request.

        prompt_token_ids:
            Complete list of prompt token IDs for this request.
            Each output carries this field to facilitate API layer returning prompt tokens in final response.
    """
    request_id: int
    text: str
    token_ids: list[int]
    finished: bool
    prompt_token_ids: list[int]
    # Per-request timing metrics (populated only when finished=True in online mode).
    ttft: Optional[float] = None       # Time to first token (seconds)
    tpot: Optional[float] = None       # Time per output token (seconds)
    total_time: Optional[float] = None # Total inference time for this request (seconds)
