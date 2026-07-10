package main

import (
	"flag"
	"fmt"
	"strconv"
	"strings"
	"time"
)

const (
	// envBotToken is the ONLY source of the Telegram bot token: it is read from
	// the environment, never accepted as a flag (so it cannot leak into shell
	// history or a process listing) and never logged.
	envBotToken = "TG2LLM_BOT_TOKEN"
	// envDraft, when truthy, puts the bot in DRAFT mode: it generates and
	// persists a reply but never delivers it (generate-but-manual-send safety).
	envDraft = "TG2LLM_BOT_DRAFT"
	// envDisableSend is the kill-switch: when truthy, sending is disabled
	// entirely regardless of DRAFT mode.
	envDisableSend = "TG2LLM_BOT_DISABLE_SEND"
)

// botConfig is the fully-resolved bot configuration. token is unexported and is
// never written to a log.
type botConfig struct {
	token            string
	dbPath           string
	llamaHost        string
	baseURL          string
	personaSystem    string
	profileHost      string
	profileModel     string
	whisperModel     string
	whisperBin       string
	whisperLang      string
	visionHost       string
	visionModel      string
	visionPrompt     string
	ownerID          int64
	allowSeed        []int64
	replyTimeout     time.Duration
	contextTurns     int
	profileEvery     int
	temperature      float64
	repeatPenalty    float64
	presencePenalty  float64
	frequencyPenalty float64
	repeatLastN      int
	splitWordProb    float64
	splitMaxMsgs     int
	draft            bool
	sendDisabled     bool
	openWhenEmpty    bool
	verbose          bool
}

// parseConfig registers the bot flags on a fresh FlagSet and reads the token
// from the environment via getenv (injected so the token source is testable).
// The token is deliberately absent from the flag set.
func parseConfig(args []string, getenv func(string) string) (botConfig, error) {
	fs := flag.NewFlagSet("bot", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(),
			"Usage: bot [flags]\n\nThe bot token is read from $%s (never a flag).\n\nFlags:\n",
			envBotToken)
		fs.PrintDefaults()
	}

	dbPath := fs.String("db", "bot_state.db", "SQLite state path (allowlist + history; empty = storeless)")
	llamaHost := fs.String("llama-host", "http://localhost:8081", "persona chat server base URL")
	baseURL := fs.String("tg-base-url", "", "Telegram Bot API base URL override (empty = https://api.telegram.org)")
	personaSystem := fs.String("persona-system", "", "persona system prompt (empty = the locked fine-tune default)")
	ownerID := fs.Int64("owner-user-id", 0, "Telegram user id that is always allowed (0 = none)")
	allow := fs.String("allow", "", "comma-separated user ids to seed into the allowlist")
	contextTurns := fs.Int("context-turns", 10, "recent conversation turns loaded as context per reply")
	profileHost := fs.String("profile-host", "", "instruct-model URL (OpenAI-compatible) that extracts the interlocutor profile (empty = profile disabled)")
	profileModel := fs.String("profile-model", "default", "model name sent to the profile extractor")
	profileEvery := fs.Int("profile-every", 15, "rebuild the interlocutor profile after this many new messages")
	whisperModel := fs.String("whisper-model", "", "whisper.cpp GGUF path for voice/video transcription (empty = disabled)")
	whisperBin := fs.String("whisper-bin", "whisper-cli", "whisper.cpp binary")
	whisperLang := fs.String("whisper-lang", "ru", "spoken language for transcription")
	visionHost := fs.String("vision-host", "", "OpenAI-compatible vision server URL for photo/video description (empty = disabled)")
	visionModel := fs.String("vision-model", "default", "vision model name sent in the request")
	visionPrompt := fs.String("vision-prompt", "", "vision prompt (empty = built-in RU prompt)")
	temperature := fs.Float64("temperature", 0.3, "persona sampling temperature (lower = less random; curbs the repetition loop)")
	repeatPenalty := fs.Float64("repeat-penalty", 1.1, "llama.cpp repeat_penalty penalising repeated tokens (1.0 = off)")
	presencePenalty := fs.Float64("presence-penalty", 0.3, "OpenAI-standard presence_penalty nudging the model off already-used tokens")
	frequencyPenalty := fs.Float64("frequency-penalty", 0.0, "OpenAI-standard frequency_penalty; scales with token frequency to break runaway repetition loops (0 = off)")
	repeatLastN := fs.Int("repeat-last-n", 0, "tokens the repetition penalties look back over; widen past llama.cpp's default 64 to catch phrases repeated several turns ago (0 = keep server default)")
	splitWordProb := fs.Float64("split-word-prob", 0.2, "probability of a message break at each inter-word gap (0 = never split a reply into multiple messages)")
	splitMaxMsgs := fs.Int("split-max-msgs", 8, "cap on messages produced by word-splitting; a reply that would exceed it is sent whole (0 = uncapped)")
	replyTimeout := fs.Duration("reply-timeout", 120*time.Second, "per-reply timeout bounding each persona model call")
	openWhenEmpty := fs.Bool("open-when-empty", false, "when the allowlist is empty, allow everyone (owner-curated bootstrap; once you /allow anyone it closes)")
	verbose := fs.Bool("verbose", false, "verbose (trace-level) logging with caller info")

	if err := fs.Parse(args); err != nil {
		return botConfig{}, err
	}

	seed, err := parseIDList(*allow)
	if err != nil {
		return botConfig{}, fmt.Errorf("invalid --allow: %w", err)
	}

	return botConfig{
		token:            getenv(envBotToken),
		dbPath:           *dbPath,
		llamaHost:        *llamaHost,
		baseURL:          *baseURL,
		personaSystem:    *personaSystem,
		profileHost:      *profileHost,
		profileModel:     *profileModel,
		whisperModel:     *whisperModel,
		whisperBin:       *whisperBin,
		whisperLang:      *whisperLang,
		visionHost:       *visionHost,
		visionModel:      *visionModel,
		visionPrompt:     *visionPrompt,
		ownerID:          *ownerID,
		allowSeed:        seed,
		replyTimeout:     *replyTimeout,
		contextTurns:     *contextTurns,
		profileEvery:     *profileEvery,
		temperature:      *temperature,
		repeatPenalty:    *repeatPenalty,
		presencePenalty:  *presencePenalty,
		frequencyPenalty: *frequencyPenalty,
		repeatLastN:      *repeatLastN,
		splitWordProb:    *splitWordProb,
		splitMaxMsgs:     *splitMaxMsgs,
		draft:            truthyEnv(getenv(envDraft)),
		sendDisabled:     truthyEnv(getenv(envDisableSend)),
		openWhenEmpty:    *openWhenEmpty,
		verbose:          *verbose,
	}, nil
}

// truthyEnv reports whether an env var is set to an affirmative value ("1",
// "true", "yes", "on"; case-insensitive). Anything else — including unset — is
// false, so both DRAFT mode and the send kill-switch default to off.
func truthyEnv(v string) bool {
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// parseIDList parses a comma-separated list of int64 user ids, trimming
// whitespace and skipping empty entries. An empty input yields no ids.
func parseIDList(s string) ([]int64, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	var ids []int64
	for _, part := range strings.Split(s, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		id, err := strconv.ParseInt(part, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("%q is not a valid user id", part)
		}
		ids = append(ids, id)
	}
	return ids, nil
}
