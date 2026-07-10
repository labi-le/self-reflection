package bot

// DefaultPersonaSystem is the LOCKED persona system prompt and the single source
// of truth shared across train and serve: the llama.cpp Brain injects it as the
// leading system turn when no persona is configured, and the chatml dataset
// builder (cmd/parse --persona-system) defaults to it, so the fine-tune sees the
// exact same system prompt the live bot serves. It MUST stay byte-identical to
// the string used during fine-tuning — changing a single rune silently changes
// the persona (a golden guard test in the llamacpp package pins these bytes).
const DefaultPersonaSystem = "Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."
