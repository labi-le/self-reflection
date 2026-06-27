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
