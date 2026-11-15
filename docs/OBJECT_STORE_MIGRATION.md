# Changing the object-store bucket or endpoint

The S3 settings live in the `object-store` integration connection (the
Integrations credentials form). llm-bawt and BawtHub both re-read that
connection at runtime, with a cache of about 60 seconds, so a saved change
takes effect without a restart.

That is safe for a **credential rotation** and unsafe for a **location
change**. Stored rows keep relative object keys, not URLs or bucket names.
Each key is resolved against whatever endpoint and bucket the connection
names *now*. Point the connection at an empty or different bucket and every
existing key misses there. Nothing copies data for you, and no ordering of
writes avoids it: a key written to the old bucket a moment before the switch
is orphaned the same way.

The same applies when no connection record exists and the legacy
`LLM_BAWT_S3_*` / `S3_*` environment variables are changed and the services
restarted.

## Which case is this?

| Change | Kind | Procedure |
|---|---|---|
| New access key or secret, same endpoint and bucket | Credential rotation | [Rotate credentials](#rotate-credentials) — live, no copy |
| Different bucket, same endpoint | Storage migration | [Migrate](#migrate-to-a-new-bucket-or-endpoint) |
| Different endpoint (new server, provider or region), even with the same bucket name | Storage migration | [Migrate](#migrate-to-a-new-bucket-or-endpoint) |
| Same server reached by a new hostname or address that serves the *same* buckets | Credential rotation in effect | Rotate, then confirm an old object reads before saving |

When in doubt, list a few known keys at the new location first. If they are
there, it is a rotation; if not, it is a migration.

## Rotate credentials

1. Create the new key on the object store with read, write, list and delete
   on the **same** bucket.
2. Save it in the `object-store` connection. Leave the old key valid.
3. Wait at least the cache window (about 60 s) for both services.
4. Open an existing image and a newly uploaded one; both should load.
5. Revoke the old key.

## Migrate to a new bucket or endpoint

### What is stored where

Everything shares one bucket, separated by prefix. Copy the **whole bucket**
with unchanged keys; do not cherry-pick prefixes.

| Prefix | Contents | Written by | Referenced from |
|---|---|---|---|
| `blobs/` | Uploaded and attached media: originals and derived variants | llm-bawt media store, BawtHub uploads | llm-bawt `media_assets` (`storage_key`, or derived from `sha256`); BawtHub `Upload` rows (derived from `sha256`) |
| `media/` | Generated videos, images and their thumbnails | llm-bawt generation | `media_generations.file_path`, `thumbnail_path` |
| `toolblobs/` | Overflow tool-call results | llm-bawt tool-call store | `tool_call_records.result_blob_key` |
| `diffs/` | Before/after bytes of changed files per turn | llm-bawt changed-files store | `turn_changed_files.before_blob_key`, `after_blob_key` |

Database columns hold the key **without** the prefix; the services add it.
The copy must therefore keep the same prefixes and paths exactly. A row that
names a key the current bucket lacks fails to load. The optional filesystem
read fallback (`LLM_BAWT_S3_FALLBACK_FS`, `S3_FALLBACK_FS`) only helps for
objects still on local disk from a filesystem-to-S3 move; it never reads the
old bucket.

### 1. Prepare

- **Back up both sides of the reference.** Back up the llm-bawt and BawtHub
  databases (their rows are the only map from data to keys) and keep the old bucket intact
  and read-only for the whole procedure. Do not delete or empty it until the
  new location has been in use and verified for a while.
- **Create the new bucket** and a key with read, write, list and delete on it.
- **Inventory the old bucket**, per prefix, and save the result:

  ```bash
  # rclone remotes "old" and "new" point at the two locations
  for p in blobs media toolblobs diffs; do
    printf '%s ' "$p"; rclone size "old:<old-bucket>/$p" --json
  done > inventory-before.txt
  ```

- **Count referenced keys** in the llm-bawt database, for the post-copy check:

  ```sql
  SELECT count(*) FROM media_generations WHERE file_path IS NOT NULL;
  SELECT count(*) FROM media_generations WHERE thumbnail_path IS NOT NULL;
  SELECT count(DISTINCT result_blob_key) FROM tool_call_records
   WHERE result_blob_key IS NOT NULL;
  SELECT count(*) FROM (
    SELECT before_blob_key k FROM turn_changed_files WHERE before_blob_key IS NOT NULL
    UNION SELECT after_blob_key FROM turn_changed_files WHERE after_blob_key IS NOT NULL
  ) s;
  SELECT count(*) FROM media_assets;
  ```

### 2. Bulk copy while live

Copying while the services keep running shortens the window. Never delete on
either side:

```bash
rclone copy --checksum "old:<old-bucket>" "new:<new-bucket>"
```

Use `copy`, not `sync`: `sync` deletes anything at the destination that is
missing from the source.

### 3. Settle writes

Writes continue during the bulk copy, and named media assets can be
**replaced** in place, so a final pass has to run with writers stopped.

1. Stop every process that writes object keys: the llm-bawt app and its
   workers (including the media and tool-result garbage collectors) and the
   BawtHub frontend services. A short maintenance window is simpler than
   trying to fence individual writers.
2. Run the copy again. It now transfers only what changed:

   ```bash
   rclone copy --checksum "old:<old-bucket>" "new:<new-bucket>"
   ```

If stopping writers is not possible, keep the old location as the source of
truth until cutover, then run step 6's delta copy from old to new with
`--ignore-existing` immediately afterwards. That covers content-addressed
objects but can miss an in-place replacement made in the gap, so the window is
the recommended path.

### 4. Verify before cutover

With writers still stopped:

```bash
rclone check --checksum --one-way "old:<old-bucket>" "new:<new-bucket>"
# Expect: 0 differences found
for p in blobs media toolblobs diffs; do
  printf '%s ' "$p"; rclone size "new:<new-bucket>/$p" --json
done > inventory-after.txt
```

Compare `inventory-after.txt` with `inventory-before.txt` (counts and bytes
equal or larger), then spot-check referenced keys from each table against the
new location, for example:

```bash
rclone lsf "new:<new-bucket>/media/<file_path from a recent media_generations row>"
rclone lsf "new:<new-bucket>/toolblobs/<a recent result_blob_key>"
rclone lsf "new:<new-bucket>/diffs/<a recent after_blob_key>"
```

Do not switch until the check is clean.

### 5. Cut over

1. Save the new endpoint, bucket and key in the `object-store` connection.
2. Start the writers again. A restart also clears the connection cache;
   otherwise allow about 60 s.
3. Verify through the product, one item per family:
   - an existing chat image and task image load, and a new upload works;
   - an existing generated video or image plays, and its thumbnail shows;
   - an older tool call with an overflowed result opens its full result;
   - an older turn's changed-file diff opens.
4. Watch the logs for object-store `404` / `NoSuchKey` errors for a while.

Record the inventories, the `rclone check` output and these results with the
change.

### 6. Rollback

After cutover, new objects exist **only** in the new bucket. Pointing the
connection back without copying them would orphan them the same way the
original problem does. To roll back:

1. Stop the writers.
2. Copy everything newer back to the old location, without deleting:

   ```bash
   rclone copy --checksum "new:<new-bucket>" "old:<old-bucket>"
   ```

3. Run `rclone check --checksum --one-way "new:<new-bucket>" "old:<old-bucket>"`
   and expect no differences.
4. Save the old endpoint, bucket and key in the connection and start the
   writers.

Keep the new bucket until the rollback is verified; it holds the only copy of
anything written while it was live.

### 7. Retire the old location

Once the new location has run cleanly for long enough that a rollback is no
longer wanted, revoke the old key and archive or delete the old bucket.

## Source map

| Concern | File |
|---|---|
| Connection resolution and 60 s cache (llm-bawt) | `src/llm_bawt/media/object_store_credentials.py` |
| Prefix per family, backend selection | `src/llm_bawt/media/object_store.py` (`_S3_KEY_PREFIXES`, `get_blob_backend`) |
| Uploaded media keys | `src/llm_bawt/media/store.py`, `src/llm_bawt/media/asset_kinds.py` |
| Generated media keys | `src/llm_bawt/media/storage.py`, `src/llm_bawt/media/db.py` |
| Tool-result overflow | `src/llm_bawt/service/tool_call_store.py` |
| Changed-file diffs | `src/llm_bawt/service/changed_files_store.py` |
| BawtHub uploads (`blobs/`, rebuilt on config change) | BawtHub `frontend/src/lib/storage.ts` (`resolveS3Config`, `getStorage`) |
