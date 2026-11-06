"""Split the command publisher's dedupe-Lua ``eval`` arguments (TASK-2120).

``eval(script, numkeys, dedupe_key, stream, *route_keys, maxlen, request_id,
*field_pairs)``: the number of route-record keys varies, so fakes split on
``numkeys`` instead of binding positions.
"""


def split_publish_eval(numkeys, args):
    """``(keys, request_id, field_pairs)``; ``keys[0]`` dedupe, ``keys[1]`` stream."""
    keys = list(args[:numkeys])
    _maxlen, request_id, *pairs = args[numkeys:]
    return keys, request_id, pairs
