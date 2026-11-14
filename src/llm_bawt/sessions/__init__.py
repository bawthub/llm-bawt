"""Session (conversation thread) domain helpers.

Threads themselves live in the ``sessions`` table (storage in
``memory/postgresql.py``, REST in ``service/routes/sessions.py``); this
package holds session-adjacent behaviors that don't belong in either.
"""
