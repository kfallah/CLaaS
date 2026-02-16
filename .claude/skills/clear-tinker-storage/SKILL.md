---
name: clear-tinker-storage
description: Delete all checkpoints from your Thinking Machine Tinker prod account to reduce storage charges. Requires TINKER_API_KEY.
---

# Clear Tinker Storage

Delete all checkpoints from the authenticated Tinker account to free up storage.

## Instructions

When this skill is invoked, perform the following steps:

1. **Load the API key** from `docker/.env.tinker` (the `TINKER_API_KEY` variable). If not found, ask the user to provide one.

2. **List all checkpoints** by paginating through the Tinker REST API:
   ```
   GET https://tinker.thinkingmachines.dev/services/tinker-prod/api/v1/checkpoints
   Headers: X-API-Key: <key>
   Params: limit=100, offset=0
   ```
   Paginate using the `cursor.total_count` field until all checkpoints are fetched.

3. **Display a summary** to the user before deleting:
   - Total number of checkpoints
   - Total storage in GB
   - Breakdown by `checkpoint_type` (training vs sampler)

4. **Ask for confirmation** before proceeding with deletion. This is a destructive, irreversible operation.

5. **Delete each checkpoint** by parsing the `tinker_path` and calling:
   ```
   DELETE https://tinker.thinkingmachines.dev/services/tinker-prod/api/v1/training_runs/{training_run_id}/checkpoints/{checkpoint_id}
   Headers: X-API-Key: <key>
   ```
   The `tinker_path` format is `tinker://{training_run_id}/{checkpoint_id}`. Split on the first `/` after removing the `tinker://` prefix to extract both parts.

6. **Verify deletion** by re-querying the list endpoint and confirming 0 checkpoints remain.

7. **Report results**: number deleted, number failed, storage freed.

## Authentication

- The API key must be a valid Tinker API key (starts with `tml-`)
- Auth header format: `X-API-Key: <key>`
- If the key returns 401, ask the user for a fresh key

## Notes

- All API calls use `httpx` with a 30-second timeout
- This deletes checkpoints across ALL training runs for the authenticated user
- There is no undo — deleted checkpoints cannot be recovered
- After clearing, new LoRAs must be re-initialized via `POST /v1/lora/init`
