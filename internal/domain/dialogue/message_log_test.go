package dialogue

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/rs/zerolog"
)

func TestMessageMarshalZerologObject(t *testing.T) {
	reply := 42
	msg := Message{
		ID:            7,
		Date:          time.Date(2026, 6, 19, 10, 0, 0, 0, time.UTC),
		Sender:        "Alice",
		MediaType:     "voice_message",
		HasPhoto:      true,
		ReplyToMsgID:  &reply,
		ForwardedFrom: "Channel",
	}
	var buf bytes.Buffer
	logger := zerolog.New(&buf)
	logger.Info().Object("msg", msg).Msg("test")

	var got map[string]any
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal log: %v", err)
	}
	obj, ok := got["msg"].(map[string]any)
	if !ok {
		t.Fatalf("msg field missing or not an object: %v", got)
	}
	if obj["id"].(float64) != 7 {
		t.Errorf("id = %v, want 7", obj["id"])
	}
	if obj["sender"] != "Alice" {
		t.Errorf("sender = %v, want Alice", obj["sender"])
	}
	if obj["media_type"] != "voice_message" {
		t.Errorf("media_type = %v, want voice_message", obj["media_type"])
	}
	if obj["has_photo"] != true {
		t.Errorf("has_photo = %v, want true", obj["has_photo"])
	}
	if obj["reply_to"].(float64) != 42 {
		t.Errorf("reply_to = %v, want 42", obj["reply_to"])
	}
	if obj["forwarded_from"] != "Channel" {
		t.Errorf("forwarded_from = %v, want Channel", obj["forwarded_from"])
	}
}

func TestMessageMarshalZerologObjectOmitsEmpty(t *testing.T) {
	msg := Message{ID: 1, Date: time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC), Sender: "Bob"}
	var buf bytes.Buffer
	logger := zerolog.New(&buf)
	logger.Info().Object("msg", msg).Msg("x")

	var got map[string]any
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal log: %v", err)
	}
	obj, ok := got["msg"].(map[string]any)
	if !ok {
		t.Fatalf("msg field missing or not an object: %v", got)
	}
	for _, k := range []string{"media_type", "has_photo", "reply_to", "forwarded_from"} {
		if _, present := obj[k]; present {
			t.Errorf("optional field %q present for bare message, want omitted", k)
		}
	}
}

// TestMessageMarshalZerologObjectIdentityFields locks the new identity/dataset
// fields (FromID, ActorID, ChatType, ChatID) into the structured log output:
// present ones must appear under their snake_case keys, absent ones must be
// OMITTED (key absent), matching the existing omit-when-empty convention.
func TestMessageMarshalZerologObjectIdentityFields(t *testing.T) {
	msg := Message{
		ID:       1,
		Date:     time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC),
		Sender:   "Alice",
		FromID:   "user1",
		ChatType: "personal_chat",
		ChatID:   "42",
		// ActorID intentionally left empty — must be omitted from the log.
	}
	var buf bytes.Buffer
	logger := zerolog.New(&buf)
	logger.Info().Object("msg", msg).Msg("x")

	var got map[string]any
	if err := json.Unmarshal(buf.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal log: %v", err)
	}
	obj, ok := got["msg"].(map[string]any)
	if !ok {
		t.Fatalf("msg field missing or not an object: %v", got)
	}
	if obj["from_id"] != "user1" {
		t.Errorf("from_id = %v, want \"user1\"", obj["from_id"])
	}
	if obj["chat_type"] != "personal_chat" {
		t.Errorf("chat_type = %v, want \"personal_chat\"", obj["chat_type"])
	}
	if obj["chat_id"] != "42" {
		t.Errorf("chat_id = %v, want \"42\"", obj["chat_id"])
	}
	// Empty ActorID must be OMITTED (key absent, not present-as-""), matching
	// the existing MediaType / ForwardedFrom style.
	if _, present := obj["actor_id"]; present {
		t.Errorf("actor_id key present for empty ActorID, want omitted")
	}

	// Also lock the raw JSON substrings the plan requires (defence-in-depth
	// against tautological map-based assertions).
	raw := buf.String()
	if !bytes.Contains([]byte(raw), []byte(`"from_id":"user1"`)) {
		t.Errorf("raw log missing \"from_id\":\"user1\": %s", raw)
	}
	if !bytes.Contains([]byte(raw), []byte(`"chat_type":"personal_chat"`)) {
		t.Errorf("raw log missing \"chat_type\":\"personal_chat\": %s", raw)
	}
	if bytes.Contains([]byte(raw), []byte(`"actor_id"`)) {
		t.Errorf("raw log contains \"actor_id\" key though ActorID is empty: %s", raw)
	}
}
