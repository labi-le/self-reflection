package app

import (
	"fmt"
	"strconv"
	"sync"

	"tg2llm/internal/domain/dialogue"
)

// ParseService orchestrates the parse pipeline: read messages, apply rules,
// optionally decode media, and write formatted output.
type ParseService struct {
	source  MessageSource
	sink    DialogueSink
	decoder MediaDecoder
	opts    ParseOptions
}

// NewParseService creates a new parse service. decoder may be nil.
func NewParseService(source MessageSource, sink DialogueSink, decoder MediaDecoder, opts ParseOptions) *ParseService {
	return &ParseService{
		source:  source,
		sink:    sink,
		decoder: decoder,
		opts:    opts,
	}
}

// Run executes the full parse pipeline. Returns an error if reading or writing fails.
func (s *ParseService) Run() error {
	messages, referencedIDs, err := s.source.ReadMessages()
	if err != nil {
		return fmt.Errorf("reading messages: %w", err)
	}

	decodedByKey := s.decodeAll(messages)

	var currentDay string

	for _, msg := range messages {
		if !s.inScope(msg) {
			continue
		}

		text := dialogue.CleanText(msg)

		// Media decoding
		if s.decoder != nil {
			if decoded := decodedByKey[decodeKey(msg)]; decoded != "" {
				if s.opts.Format == "chatml" {
					// The chatml sink builds turns from ExtractedText and skips empty
					// ones, so without this fold every voice message and video note
					// silently vanishes from the corpus. The rendering depends on the
					// side: a turn the persona reads must look like the bot's inference
					// input, a turn it wrote is the supervised target and may carry only
					// what the human actually said.
					if msg.FromID == s.opts.OwnerFromID {
						msg.ExtractedText = dialogue.PersonaOwnText(msg, decoded)
					} else {
						msg.ExtractedText = dialogue.PersonaText(msg, decoded)
					}
					text = msg.ExtractedText
				} else {
					text = dialogue.FormatDecoded(msg, decoded)
					if caption := dialogue.Caption(msg); caption != "" {
						text = caption + " " + text
					}
				}
			}
		}

		// Forward wrapping (after media decode)
		forwardID := msg.ForwardedFromID
		if forwardID == "" {
			forwardID = msg.ForwardedFrom
		}
		if forwardID != "" {
			sourceName := msg.ForwardedFrom
			if sourceName == "" {
				sourceName = "Unknown"
			}
			text = "[Forwarded from " + sourceName + "]: " + text
		}

		// Drop rule
		if dialogue.IsUseless(text) && !referencedIDs[msg.ID] {
			continue
		}

		// Day header
		if s.opts.ShowDateHeaders {
			day := msg.Date.Format("2006-01-02")
			if day != currentDay {
				if err := s.sink.WriteDayHeader(day); err != nil {
					return fmt.Errorf("writing day header: %w", err)
				}
				currentDay = day
			}
		}

		if err := s.sink.WriteMessage(msg, text); err != nil {
			return fmt.Errorf("writing message: %w", err)
		}
	}

	if err := s.sink.Close(); err != nil {
		return err
	}

	if s.decoder != nil {
		if err := s.decoder.Flush(); err != nil {
			return fmt.Errorf("flushing decoded media: %w", err)
		}
	}

	return nil
}

// chatInScope reports whether a message's chat type passes the --chat-scope
// filter. "personal" keeps only 1:1 personal chats; any other value ("all")
// keeps every chat type. It is consulted only in chatml mode.
func chatInScope(scope, chatType string) bool {
	if scope == "personal" {
		return chatType == "personal_chat"
	}
	return true
}

// inScope reports whether a message survives the date filter and — in chatml mode
// only — the --chat-scope filter. Gated on Format because ChatScope defaults to
// "personal" while the text pipeline has no scope concept and must keep emitting
// every chat. Run and decodeAll share it: decoding a message the run then drops
// would spend a whisper or vision call on nothing (a full-account export carries
// tens of thousands of them).
func (s *ParseService) inScope(msg dialogue.Message) bool {
	if !s.opts.DateRange.Contains(msg.Date) {
		return false
	}
	return s.opts.Format != "chatml" || chatInScope(s.opts.ChatScope, msg.ChatType)
}

// decodeAll decodes media for every in-scope message, using s.opts.Jobs worker
// goroutines (Jobs <= 1 stays sequential). Results are keyed by decodeKey (chat id
// + message id) and the caller writes them back in original order, so any worker
// count produces byte-identical output. The collecting goroutine is the sole
// writer of the returned map, so no locking is needed around it.
func (s *ParseService) decodeAll(messages []dialogue.Message) map[string]string {
	decoded := make(map[string]string)
	if s.decoder == nil {
		return decoded
	}

	jobs := s.opts.Jobs
	if jobs < 1 {
		jobs = 1
	}

	if jobs == 1 {
		for _, msg := range messages {
			if s.inScope(msg) {
				if text := s.decoder.Decode(msg); text != "" {
					decoded[decodeKey(msg)] = text
				}
			}
		}
		return decoded
	}

	type result struct {
		key  string
		text string
	}
	jobCh := make(chan dialogue.Message)
	resCh := make(chan result)

	var wg sync.WaitGroup
	for i := 0; i < jobs; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for msg := range jobCh {
				if text := s.decoder.Decode(msg); text != "" {
					resCh <- result{key: decodeKey(msg), text: text}
				}
			}
		}()
	}

	go func() {
		for _, msg := range messages {
			if s.inScope(msg) {
				jobCh <- msg
			}
		}
		close(jobCh)
	}()

	go func() {
		wg.Wait()
		close(resCh)
	}()

	for r := range resCh {
		decoded[r.key] = r.text
	}
	return decoded
}

// decodeKey identifies a message across the whole run. A Telegram message id is
// unique only within its chat, and a full-account export flattens every chat into
// one slice, so keying decodes by id alone lets one chat's transcript surface on
// another chat's message — in chatml mode, straight into a training turn.
func decodeKey(msg dialogue.Message) string {
	return msg.ChatID + ":" + strconv.Itoa(msg.ID)
}
