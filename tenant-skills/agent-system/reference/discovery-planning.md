# Discovery and planning

Start with the requested outcome and inspect enough code to choose a bounded approach. Prefer the existing architecture and avoid introducing duplication.

Refactor first when it is necessary for correctness, testability, or keeping the change small. File size is a warning signal, not an automatic prerequisite. A cohesive large module can receive a small fix. Choose composition, a helper, or an existing abstraction according to the problem; an ABC is not mandatory.

Record substantial tradeoffs briefly. The owner can make ordinary design and task-splitting decisions. Ask Gavel only about a real integration conflict, and Nick about material product/scope or irreversible decisions.

Do not make unrelated cleanup or future generality a dependency. For maintenance, compare an offline procedure with new online machinery before expanding scope.
