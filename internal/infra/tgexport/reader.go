package tgexport

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/domain/dialogue"
	"tg2llm/pkg/ctxlog"
)

// Reader implements app.MessageSource for Telegram JSON exports. It handles both
// the single-chat export (top-level "messages" or a bare array) and the
// full-account export (root object with personal_information/chats/left_chats).
type Reader struct {
	path      string
	logger    zerolog.Logger
	ownerHint string
	owner     string
}

// NewReader creates a Reader for the given JSON export file.
func NewReader(path string, logger zerolog.Logger) *Reader {
	return &Reader{path: path, logger: logger}
}

// WithOwnerHint sets the explicit account-owner id (e.g. from --owner-user-id)
// that wins over auto-detection. It returns the reader for fluent construction.
func (r *Reader) WithOwnerHint(hint string) *Reader {
	r.ownerHint = hint
	return r
}

// OwnerFromID returns the account owner resolved by the most recent ReadMessages
// call. It is empty when no owner was needed or resolvable (single-chat / legacy
// exports without identity data).
func (r *Reader) OwnerFromID() string {
	return r.owner
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

	c := newCollector(log)

	root, isObject := raw.(map[string]interface{})
	if isObject && isFullAccount(root) {
		for _, chat := range fullAccountChats(root) {
			rawMsgs, _ := chat["messages"].([]interface{})
			c.collect(getString(chat, "type"), numberToString(chat["id"]), rawMsgs)
		}
		owner, err := r.detectOwnerFullAccount(root)
		if err != nil {
			return nil, nil, err
		}
		r.owner = owner
	} else {
		chatType, chatID, rawMsgs := singleChat(raw)
		c.collect(chatType, chatID, rawMsgs)
		r.owner = r.detectOwnerSingleChat(chatType, chatID, c.messages)
	}

	if c.skipped > 0 {
		log.Warn().Int("skipped", c.skipped).Msg("skipped non-message or unparseable records")
	}

	return c.messages, c.referencedIDs, nil
}

type collector struct {
	referencedIDs map[int]bool
	messages      []dialogue.Message
	skipped       int
	log           zerolog.Logger
}

func newCollector(log zerolog.Logger) *collector {
	return &collector{referencedIDs: make(map[int]bool), log: log}
}

// collect records reply_to targets from ALL records (service included, so the
// keep-if-referenced drop rule still sees them) but appends only message-type
// records, tagged with chatType / chatID.
func (c *collector) collect(chatType, chatID string, rawMsgs []interface{}) {
	for _, rm := range rawMsgs {
		m, ok := rm.(map[string]interface{})
		if !ok {
			continue
		}

		if rid := getInt(m, "reply_to_message_id"); rid != nil {
			c.referencedIDs[*rid] = true
		}

		if getString(m, "type") != "message" {
			c.skipped++
			c.log.Debug().Int("id", getIntVal(m, "id")).Str("reason", "non-message").Msg("skipping record")
			continue
		}

		msgDate, ok := resolveDate(m)
		if !ok {
			c.skipped++
			c.log.Debug().Int("id", getIntVal(m, "id")).Str("reason", "undated").Msg("skipping record")
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
			FromID:          getString(m, "from_id"),
			ActorID:         getString(m, "actor_id"),
			ChatType:        chatType,
			ChatID:          chatID,
		}

		if _, hasPhoto := m["photo"]; hasPhoto {
			msg.HasPhoto = true
		}
		if msg.Sender == "" {
			msg.Sender = "Unknown"
		}

		c.messages = append(c.messages, msg)
	}
}

var dateLayouts = [...]string{
	"2006-01-02T15:04:05",
	"2006-01-02 15:04:05",
	time.RFC3339,
	"2006-01-02T15:04",
	"2006-01-02 15:04",
	"2006-01-02",
}

// resolveDate parses the local-ISO "date" string (six layouts); when it is empty
// or unparseable it falls back to the "date_unixtime" seconds string. ok is false
// only when neither yields a timestamp.
func resolveDate(m map[string]interface{}) (time.Time, bool) {
	if s := getString(m, "date"); s != "" {
		for _, layout := range dateLayouts {
			if t, err := time.Parse(layout, s); err == nil {
				return t, true
			}
		}
	}
	if s := getString(m, "date_unixtime"); s != "" {
		if n, err := strconv.ParseInt(s, 10, 64); err == nil {
			return time.Unix(n, 0).UTC(), true
		}
	}
	return time.Time{}, false
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

// numberToString renders a JSON id (chat id / user_id) as an exact integer
// string. JSON numbers arrive as float64; every Telegram peer id is below 2^53,
// so float64->int64 is lossless — and it never passes through the narrower int
// that getInt uses.
func numberToString(v interface{}) string {
	switch n := v.(type) {
	case string:
		return n
	case float64:
		return strconv.FormatInt(int64(n), 10)
	case json.Number:
		return n.String()
	default:
		return ""
	}
}
