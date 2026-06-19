# KV Cache Block Manager

## Source Modules

- `babyvllm/engine/block_manager.py`
- `babyvllm/engine/sequence.py`
- `babyvllm/engine/model_runner.py`

`BlockManager` maps each logical sequence block to a physical KV-cache block. `Sequence.block_table` stores those physical block ids, while the model runner uses the block table to build attention metadata.

## Block State

```mermaid
stateDiagram-v2
  [*] --> Free: "Block id in free_block_ids"
  Free --> UsedPrivate: "_allocate_block(), ref_count=1"
  UsedPrivate --> Shared: "Prefix cache hit while already used, ref_count++"
  Shared --> UsedPrivate: "deallocate one sequence, ref_count--"
  UsedPrivate --> Free: "ref_count becomes 0, _deallocate_block()"
  Free --> UsedPrivate: "Cache hit for old block id, _allocate_block(block_id)"
```

`hash_to_block_id` is independent of the live/free state. A block can be free but still remembered as the storage location for a full historical token block. On a later prefix-cache hit, the manager can allocate that same block id again.

## Chunk Allocation

```mermaid
flowchart TD
  Start["allocate_chunk(seq, chunk_size)"] --> Range["Compute token range: start=num_computed_tokens, end=start+chunk_size"]
  Range --> Blocks["Find logical block range covering the chunk"]
  Blocks --> Loop["For each logical block"]
  Loop --> NewBlock{"i >= len(seq.block_table)?"}
  NewBlock -->|No| AppendExisting["Update existing physical block"]
  AppendExisting --> MaybeHash{"Block becomes full?"}
  MaybeHash -->|Yes| Register["hash_to_block_id[hash] = block_id"]
  MaybeHash -->|No| Next
  Register --> Next["Next logical block"]

  NewBlock -->|Yes| Full{"Will this logical block be full?"}
  Full -->|No| NewPartial["Allocate free block, hash=-1"]
  NewPartial --> AppendTable["Append physical block id to seq.block_table"]
  Full -->|Yes| Hash["Compute chained hash from previous full block hash"]
  Hash --> Hit{"hash_to_block_id hit and token_ids match?"}
  Hit -->|Yes| Reuse["Reuse cached block, increment or reallocate ref_count"]
  Hit -->|No| Miss["Allocate free block, write hash and token_ids"]
  Reuse --> Cached["seq.num_cached_tokens += block_size"]
  Cached --> AppendTable
  Miss --> AppendTable
  AppendTable --> Next
```

Only full blocks are inserted into `hash_to_block_id`. Partial blocks keep `hash=-1`, which avoids false prefix-cache hits and reduces hash-table churn.

## Allocation Capacity

```mermaid
flowchart TD
  Can["can_allocate_chunk(seq, chunk_size)"] --> Current["current_blocks = len(seq.block_table)"]
  Current --> Target["target_blocks = ceil((num_computed_tokens + chunk_size) / block_size)"]
  Target --> Needed["num_new_blocks_needed = target_blocks - current_blocks"]
  Needed --> Compare{"needed <= len(free_block_ids)?"}
  Compare -->|Yes| OK["Chunk can be allocated"]
  Compare -->|No| No["Scheduler must reduce chunk or preempt"]
```

The scheduler uses this check in two ways: Decode loops may preempt another running sequence until a one-token extension fits, while Prefill uses `_get_max_chunk_size()` to shrink the chunk to the available block budget.

## Deallocation

```mermaid
flowchart TD
  Dealloc["deallocate(seq)"] --> Loop["For block_id in seq.block_table"]
  Loop --> Dec["blocks[block_id].ref_count -= 1"]
  Dec --> Zero{"ref_count == 0?"}
  Zero -->|Yes| Free["_deallocate_block(): clear token_ids, move used -> free"]
  Zero -->|No| Shared["Another sequence still references cached prefix"]
  Free --> Continue["Continue"]
  Shared --> Continue
  Continue --> Done{"All blocks processed?"}
  Done -->|No| Loop
  Done -->|Yes| ResetSeq["seq.block_table = [], seq.num_cached_tokens = 0"]
```

Deallocation does not remove full-block hashes from `hash_to_block_id`. That is intentional: the mapping records where a reusable full prefix block was stored, even if no active sequence currently references it.

