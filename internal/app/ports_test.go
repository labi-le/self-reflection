package app_test

import (
	"bytes"
	"encoding/json"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/app"
	"tg2llm/internal/domain/dialogue"
)

func TestParseOptionsMarshalZerologObject(t *testing.T) {
	from := time.Date(2026, 6, 19, 0, 0, 0, 0, time.UTC)
	to := time.Date(2026, 6, 20, 23, 59, 59, 0, time.UTC)
	opts := app.ParseOptions{
		ShowDateHeaders: true,
		Jobs:            4,
		DateRange:       dialogue.DateRange{From: &from, To: &to},
	}
	var buf bytes.Buffer
	logger := zerolog.New(&buf)
	logger.Info().Object("opts", opts).Msg("x")

	obj := unmarshalObject(t, buf.Bytes(), "opts")
	if obj["show_date_headers"] != true {
		t.Errorf("show_date_headers = %v, want true", obj["show_date_headers"])
	}
	if obj["jobs"].(float64) != 4 {
		t.Errorf("jobs = %v, want 4", obj["jobs"])
	}
	if _, ok := obj["date_from"]; !ok {
		t.Error("date_from missing, want present")
	}
	if _, ok := obj["date_to"]; !ok {
		t.Error("date_to missing, want present")
	}
}

func TestParseOptionsMarshalOmitsNilDates(t *testing.T) {
	var buf bytes.Buffer
	logger := zerolog.New(&buf)
	logger.Info().Object("opts", app.ParseOptions{Jobs: 1}).Msg("x")

	obj := unmarshalObject(t, buf.Bytes(), "opts")
	for _, k := range []string{"date_from", "date_to"} {
		if _, present := obj[k]; present {
			t.Errorf("nil-date field %q present, want omitted", k)
		}
	}
}

func unmarshalObject(t *testing.T, raw []byte, key string) map[string]any {
	t.Helper()
	var got map[string]any
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatalf("unmarshal log: %v", err)
	}
	obj, ok := got[key].(map[string]any)
	if !ok {
		t.Fatalf("field %q missing or not an object: %v", key, got)
	}
	return obj
}
