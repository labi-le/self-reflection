// Command analyze turns a Telegram export into a structured psychological
// portrait and answers grounded questions about the conversation. It dispatches
// on the first argument to a per-verb FlagSet (flags precede the positional
// path within each subcommand), mirroring the repo's one-binary-per-command
// convention.
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"time"

	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/media"
	"tg2llm/internal/infra/tgexport"

	"github.com/rs/zerolog"
)

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	args := os.Args[2:]
	var err error
	switch os.Args[1] {
	case "portrait":
		err = runPortrait(args)
	case "ask":
		err = runAsk(args)
	case "participants":
		err = runParticipants(args)
	case "chats":
		err = runChats(args)
	case "-h", "--help", "help":
		usage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n", os.Args[1])
		usage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprint(os.Stderr, `Usage: analyze <command> [flags] <export.json>

Commands:
  portrait       build a structured psychological portrait (portrait.json)
  ask            interactive Q&A grounded in the conversation
  participants   list sender ids and message counts
  chats          list chats (chat_id, type, participants) to pick --chat

Flags precede the positional path within each command. Run "analyze <command> -h".
`)
}

// initLogger mirrors cmd/parse's console logger (own copy — the helper lives in
// package main there and cannot be imported; per-binary duplication is house
// style).
func initLogger(verbose bool) zerolog.Logger {
	output := zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}
	if verbose {
		zerolog.CallerMarshalFunc = func(_ uintptr, file string, line int) string {
			return filepath.Base(file) + ":" + strconv.Itoa(line)
		}
		return zerolog.New(output).Level(zerolog.TraceLevel).With().Timestamp().Caller().Logger()
	}
	return zerolog.New(output).Level(zerolog.InfoLevel).With().Timestamp().Logger()
}

func abspath(p string) string {
	if p == "" {
		return p
	}
	if abs, err := filepath.Abs(p); err == nil {
		return abs
	}
	return p
}

// warnRemote prints an egress warning when any configured host is non-local:
// raw messages and inferences about a real person then leave this device.
func warnRemote(logger zerolog.Logger, hosts ...string) {
	for _, h := range hosts {
		if h != "" && !isLocalHost(h) {
			logger.Warn().Str("host", h).Msg("remote host configured: raw messages and psychological inferences leave this device")
		}
	}
}

func isLocalHost(host string) bool {
	return stringContainsAny(host, "localhost", "127.0.0.1", "::1", "0.0.0.0")
}

func stringContainsAny(s string, subs ...string) bool {
	for _, sub := range subs {
		if len(sub) <= len(s) && indexOf(s, sub) >= 0 {
			return true
		}
	}
	return false
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// dateRangeFrom builds a DateRange from CLI strings using the shared domain parser.
func dateRangeFrom(from, to string) (dialogue.DateRange, error) {
	var dr dialogue.DateRange
	if from != "" {
		t, err := dialogue.ParseBound(from, false)
		if err != nil {
			return dr, fmt.Errorf("invalid --date-from: %w", err)
		}
		dr.From = &t
	}
	if to != "" {
		t, err := dialogue.ParseBound(to, true)
		if err != nil {
			return dr, fmt.Errorf("invalid --date-to: %w", err)
		}
		dr.To = &t
	}
	return dr, nil
}

// readMessages reads the export and returns the messages plus the resolved
// account owner (empty for legacy single-chat exports).
func readMessages(path string, logger zerolog.Logger) ([]dialogue.Message, string, error) {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return nil, "", fmt.Errorf("input file not found: %s", path)
	}
	reader := tgexport.NewReader(path, logger)
	msgs, _, err := reader.ReadMessages()
	if err != nil {
		return nil, "", fmt.Errorf("reading export: %w", err)
	}
	return msgs, reader.OwnerFromID(), nil
}

func filterByDate(msgs []dialogue.Message, dr dialogue.DateRange) []dialogue.Message {
	if dr.From == nil && dr.To == nil {
		return msgs
	}
	out := msgs[:0:0]
	for _, m := range msgs {
		if dr.Contains(m.Date) {
			out = append(out, m)
		}
	}
	return out
}

// inferTarget picks the subject when --target is omitted: the non-owner sender
// with the most messages (falling back to the overall most frequent sender).
func inferTarget(msgs []dialogue.Message, owner string) string {
	counts := map[string]int{}
	for _, m := range msgs {
		if m.FromID != "" {
			counts[m.FromID]++
		}
	}
	best, bestN := "", -1
	for id, n := range counts {
		if id == owner {
			continue
		}
		if n > bestN || (n == bestN && id < best) {
			best, bestN = id, n
		}
	}
	if best != "" {
		return best
	}
	// Only the owner spoke (or owner unknown): fall back to the top sender.
	for id, n := range counts {
		if n > bestN || (n == bestN && id < best) {
			best, bestN = id, n
		}
	}
	return best
}

// decodeMessages runs the media decoder over each message, folding any
// transcript/description into ExtractedText so it becomes analyzable signal.
func decodeMessages(msgs []dialogue.Message, dec *media.Decoder, logger zerolog.Logger) []dialogue.Message {
	for i := range msgs {
		decoded := dec.Decode(msgs[i])
		if decoded == "" {
			continue
		}
		if msgs[i].ExtractedText == "" {
			msgs[i].ExtractedText = decoded
		} else {
			msgs[i].ExtractedText = msgs[i].ExtractedText + " " + decoded
		}
	}
	if err := dec.Flush(); err != nil {
		logger.Warn().Err(err).Msg("flushing media cache")
	}
	return msgs
}

// sortedSenders returns from_ids ordered by descending message count (ties by id).
func sortedSenders(stats portrait.Stats) []string {
	ids := make([]string, 0, len(stats.PerSender))
	for id := range stats.PerSender {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		a, b := stats.PerSender[ids[i]], stats.PerSender[ids[j]]
		if a.Messages != b.Messages {
			return a.Messages > b.Messages
		}
		return ids[i] < ids[j]
	})
	return ids
}

// buildDecoder assembles the media decoder for --decode-media, mirroring
// cmd/parse's wiring: default cache beside the export, ~ expansion of the
// whisper model, and a warning when the voice model is unset.
func buildDecoder(input, cachePath, whisperModel, whisperBin, whisperLang, visionModel, visionHost, visionPrompt string, logger zerolog.Logger) *media.Decoder {
	exportRoot := filepath.Dir(abspath(input))
	if cachePath == "" {
		cachePath = filepath.Join(exportRoot, ".tg2llm_media_cache.db")
	}
	if whisperModel == "" {
		logger.Warn().Msg("whisper model not set; voice messages stay undecoded")
	} else if whisperModel[0] == '~' {
		if home, err := os.UserHomeDir(); err == nil {
			whisperModel = filepath.Join(home, whisperModel[1:])
		}
	}
	return media.BuildDefaultDecoder(exportRoot, cachePath,
		&media.WhisperConfig{Model: whisperModel, Bin: whisperBin, Lang: whisperLang},
		&media.VisionConfig{Model: visionModel, Host: visionHost, Prompt: visionPrompt},
		logger)
}

// filterByChat keeps only messages in the given chat_id. Empty chatID is a no-op.
func filterByChat(msgs []dialogue.Message, chatID string) []dialogue.Message {
	if chatID == "" {
		return msgs
	}
	out := msgs[:0:0]
	for _, m := range msgs {
		if m.ChatID == chatID {
			out = append(out, m)
		}
	}
	return out
}
