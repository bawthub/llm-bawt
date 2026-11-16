"""Exact original-entry transport disposition, never execution permission."""

_SOURCE = """
local stream, receipt_key = KEYS[1], KEYS[2]
local group, id, expected = ARGV[1], ARGV[2], ARGV[3]
local receipt = cjson.decode(expected)
if receipt.version ~= 1 or receipt.state ~= 'ACK_READY'
    or receipt.stream ~= stream or receipt.group ~= group
    or receipt.entry_id ~= id or receipt.action ~= 'chat.send'
    or receipt.reason ~= 'claude_run_send_finally_v1'
    or type(receipt.boot_id) ~= 'string' or receipt.boot_id == '' then
    return -1
end
local entries = redis.call('XRANGE', stream, id, id, 'COUNT', 1)
-- -3: source gone (trimmed/deleted); only the original handler may still ACK.
if #entries ~= 1 then return -3 end
local raw, fields, keys = entries[1][2], {}, {}
for i = 1, #raw, 2 do
    if fields[raw[i]] ~= nil then return -1 end
    fields[raw[i]] = raw[i + 1]
    table.insert(keys, raw[i])
end
table.sort(keys)
local chunks = {}
for _, key in ipairs(keys) do
    local value = fields[key]
    table.insert(chunks, #key .. ':' .. key .. #value .. ':' .. value)
end
if redis.sha1hex(table.concat(chunks)) ~= receipt.source_digest
    or fields.action ~= receipt.action or fields.backend ~= receipt.backend
    or fields.request_id ~= receipt.request_id
    or (fields.session_key or '') ~= receipt.session_key
    or (fields.house_epoch or '') ~= receipt.house_epoch then
    return -1
end
-- NOGROUP is an error, never permission to mint, delete or recreate.
local pending = redis.call('XPENDING', stream, group, id, id, 1)
"""

MINT_ACK = _SOURCE + """
local prior = redis.call('GET', receipt_key)
if prior and prior ~= expected then return -1 end
if #pending == 0 then return 2 end
redis.call('SET', receipt_key, expected)
return 1
"""

SETTLE_ACK = """
-- A conflicting receipt never grants authority. A missing one can only prove
-- idempotent completion if this exact original entry is no longer pending.
local prior = redis.call('GET', KEYS[2])
if prior and prior ~= ARGV[3] then return 0 end
""" + _SOURCE + """
if not prior then
    if #pending == 0 then return 2 end
    return 0
end
redis.call('XACK', stream, group, id)
redis.call('DEL', receipt_key)
return 1
"""
