package app

import (
	"fmt"
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

	decodedByID := s.decodeAll(messages)

	var currentDay string

	for _, msg := range messages {
		if !s.opts.DateRange.Contains(msg.Date) {
			continue
		}

		text := dialogue.CleanText(msg)

		// Media decoding
		if s.decoder != nil {
			decoded := decodedByID[msg.ID]
			if decoded != "" {
				label := msg.MediaType
				if label == "" && msg.HasPhoto {
					label = "photo"
				}
				if label == "" {
					label = "media"
				}
				caption := dialogue.Caption(msg)
				tag := "[" + label + ": " + decoded + "]"
				if caption != "" {
					text = caption + " " + tag
				} else {
					text = tag
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

	return s.sink.Close()
}

// decodeAll decodes media for every in-range message, using s.opts.Jobs worker
// goroutines (Jobs <= 1 stays sequential). Results are keyed by message ID and
// the caller writes them back in original order, so any worker count produces
// byte-identical output. The collecting goroutine is the sole writer of the
// returned map, so no locking is needed around it.
func (s *ParseService) decodeAll(messages []dialogue.Message) map[int]string {
	decoded := make(map[int]string)
	if s.decoder == nil {
		return decoded
	}

	jobs := s.opts.Jobs
	if jobs < 1 {
		jobs = 1
	}

	if jobs == 1 {
		for _, msg := range messages {
			if s.opts.DateRange.Contains(msg.Date) {
				if text := s.decoder.Decode(msg); text != "" {
					decoded[msg.ID] = text
				}
			}
		}
		return decoded
	}

	type result struct {
		id   int
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
					resCh <- result{id: msg.ID, text: text}
				}
			}
		}()
	}

	go func() {
		for _, msg := range messages {
			if s.opts.DateRange.Contains(msg.Date) {
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
		decoded[r.id] = r.text
	}
	return decoded
}
