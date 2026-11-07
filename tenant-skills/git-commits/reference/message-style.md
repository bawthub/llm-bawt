# Commit messages

Use `TASK-N type(scope): concise outcome`, with a subject around 70 characters. Explain a material reason or tradeoff in the body when useful.

Include one trailer: `Co-Authored-By: <bot> (<harness>) <<bot>@bots.bawt.local>`. Use the actual identity; operator-origin work can be attributed explicitly rather than inventing a bot.

Write multiline messages to a file and pass `-F`; do not interpolate them into shell commands. Verify the resulting commit and include only the intended paths.
