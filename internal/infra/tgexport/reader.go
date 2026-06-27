package tgexport

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/domain/dialogue"
	"tg2llm/pkg/ctxlog"
)

// Reader implements app.MessageSource for Telegram JSON exports.
type Reader struct {
	path   string
	logger zerolog.Logger
}

// NewReader creates a Reader for the given JSON export file.
func NewReader(path string, logger zerolog.Logger) *Reader {
	return &Reader{path: path, logger: logger}
}

// ReadMessages reads the JSON export and returns domain messages + referenced IDs.
func (r *Reader) ReadMessages() ([]dialogue.Message, map[int]bool, error) {
	log := ctxlog.Op(r.logger, "Reader.ReadMessages")

	data, err := os.ReadFile(r.path)
	if err != nil {
		return nil, nil, fmt.Errorf("reading file: %w", err)
	}

	var raw interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, nil, fmt.Errorf("parsing JSON: %w", err)
	}

	// Get the messages array: if top-level is object, use its "messages" field;
	// if it's already an array, use it directly.
	var rawMsgs []interface{}
	switch v := raw.(type) {
	case map[string]interface{}:
		if arr, ok := v["messages"].([]interface{}); ok {
			rawMsgs = arr
		}
	case []interface{}:
		rawMsgs = v
	}

	// Build referenced_ids from ALL message objects (not filtered by type).
	referencedIDs := make(map[int]bool)
	for _, rm := range rawMsgs {
		m, ok := rm.(map[string]interface{})
		if !ok {
			continue
		}
		if rid := getInt(m, "reply_to_message_id"); rid != nil {
			referencedIDs[*rid] = true
		}
	}

	skipped := 0
	var messages []dialogue.Message
	for _, rm := range rawMsgs {
		m, ok := rm.(map[string]interface{})
		if !ok {
			continue
		}

		// Skip entries that are not chat messages (service/system events).
		if getString(m, "type") != "message" {
			skipped++
			log.Debug().Int("id", getIntVal(m, "id")).Str("reason", "non-message").Msg("skipping record")
			continue
		}

		dateStr := getString(m, "date")
		if dateStr == "" {
			skipped++
			log.Debug().Int("id", getIntVal(m, "id")).Str("reason", "empty-date").Msg("skipping record")
			continue
		}
		var msgDate time.Time
		var err error
		for _, layout := range []string{
			"2006-01-02T15:04:05",
			"2006-01-02 15:04:05",
			time.RFC3339,
			"2006-01-02T15:04",
			"2006-01-02 15:04",
			"2006-01-02",
		} {
			msgDate, err = time.Parse(layout, dateStr)
			if err == nil {
				break
			}
		}
		if err != nil {
			skipped++
			log.Debug().Int("id", getIntVal(m, "id")).Str("reason", "unparseable-date").Msg("skipping record")
			continue
		}

		msg := dialogue.Message{
			ID:              getIntVal(m, "id"),
			Date:            msgDate,
			Sender:          getString(m, "from"),
			ExtractedText:   extractText(m["text"]),
			ReplyToMsgID:    getInt(m, "reply_to_message_id"),
			ForwardedFromID: getString(m, "forwarded_from_id"),
			ForwardedFrom:   getString(m, "forwarded_from"),
			MediaType:       getString(m, "media_type"),
			StickerEmoji:    getString(m, "sticker_emoji"),
			PhotoPath:       getString(m, "photo"),
			FilePath:        getString(m, "file"),
		}

		if _, hasPhoto := m["photo"]; hasPhoto {
			msg.HasPhoto = true
		}
		if msg.Sender == "" {
			msg.Sender = "Unknown"
		}

		messages = append(messages, msg)
	}

	if skipped > 0 {
		log.Warn().Int("skipped", skipped).Msg("skipped non-message or unparseable records")
	}

	return messages, referencedIDs, nil
}

// extractText concatenates polymorphic text fields.
func extractText(textField interface{}) string {
	switch v := textField.(type) {
	case string:
		return v
	case []interface{}:
		var result string
		for _, item := range v {
			switch it := item.(type) {
			case string:
				result += it
			case map[string]interface{}:
				if t, ok := it["text"].(string); ok {
					result += t
				}
			}
		}
		return result
	default:
		return ""
	}
}

// getString safely extracts a string value from a map.
func getString(m map[string]interface{}, key string) string {
	v, ok := m[key]
	if !ok {
		return ""
	}
	s, ok := v.(string)
	if !ok {
		return ""
	}
	return s
}

// getInt safely extracts an int value from a numeric JSON field. Returns nil if absent or not a number.
func getInt(m map[string]interface{}, key string) *int {
	v, ok := m[key]
	if !ok || v == nil {
		return nil
	}
	switch n := v.(type) {
	case float64:
		i := int(n)
		return &i
	default:
		return nil
	}
}

// getIntVal returns the int value or 0.
func getIntVal(m map[string]interface{}, key string) int {
	if p := getInt(m, key); p != nil {
		return *p
	}
	return 0
}
