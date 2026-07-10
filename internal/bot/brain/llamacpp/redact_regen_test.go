package llamacpp

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"tg2llm/internal/bot"
	"tg2llm/internal/redact"
)

// piiReply carries a Russian phone number redact.Redact detects, so it is the
// canonical "dirty" completion for the PII-regeneration tests.
const piiReply = "мой номер +79001234567"

// (a) First completion has PII, the retry is clean: Reply returns the CLEAN
// reply and hits the server exactly twice (initial + one regeneration).
func Test_Reply_regenerates_until_clean_when_first_reply_has_PII(t *testing.T) {
	// Given a server that returns PII first, then a clean reply
	clean := "давай встретимся в парке в субботу"
	f := &fakePersona{replies: []string{piiReply, clean}}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "", WithTimeout(2*time.Second))

	// When
	got, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "где встретимся?"}})

	// Then the clean regeneration is returned, PII-free, after exactly two calls
	if err != nil {
		t.Fatalf("Reply: %v", err)
	}
	if got != clean {
		t.Fatalf("Reply = %q, want the clean regeneration %q", got, clean)
	}
	if redact.Redact(got) != got {
		t.Fatalf("Reply returned text still containing PII: %q", got)
	}
	if n := f.callCount(); n != 2 {
		t.Fatalf("server calls = %d, want 2 (initial + 1 retry)", n)
	}
}

// (b) Every completion has PII: after N retries Reply falls back to redaction —
// the result carries redact.Placeholder, no raw PII, and the server was hit 1+N
// times. N is set via WithMaxRedactRetries to a value distinct from the default,
// so exactly 3 calls also proves the option is honoured.
func Test_Reply_redacts_when_PII_persists_across_all_retries(t *testing.T) {
	// Given a server whose every reply contains PII, and N=2 retries
	const retries = 2
	f := &fakePersona{reply: piiReply}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "", WithTimeout(2*time.Second), WithMaxRedactRetries(retries))

	// When
	got, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "номер?"}})

	// Then the reply is redacted (placeholder present, raw PII absent) after 1+N calls
	if err != nil {
		t.Fatalf("Reply: %v", err)
	}
	if !strings.Contains(got, redact.Placeholder) {
		t.Fatalf("Reply = %q, want it to contain the redaction placeholder %q", got, redact.Placeholder)
	}
	if strings.Contains(got, "+79001234567") {
		t.Fatalf("Reply leaked raw PII: %q", got)
	}
	if redact.Redact(got) != got {
		t.Fatalf("Reply is not fully clean: %q", got)
	}
	if n, want := f.callCount(), 1+retries; n != want {
		t.Fatalf("server calls = %d, want %d (initial + %d retries)", n, want, retries)
	}
}

// (c) A clean first completion is returned unchanged with a single model call —
// the PII path adds no cost to the common case.
func Test_Reply_returns_clean_first_reply_without_regenerating(t *testing.T) {
	// Given a server that returns a clean reply
	clean := "ага, до встречи вечером"
	f := &fakePersona{reply: clean}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "", WithTimeout(2*time.Second))

	// When
	got, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "пока?"}})

	// Then the reply is returned as-is after a single call
	if err != nil {
		t.Fatalf("Reply: %v", err)
	}
	if got != clean {
		t.Fatalf("Reply = %q, want the clean reply unchanged %q", got, clean)
	}
	if n := f.callCount(); n != 1 {
		t.Fatalf("server calls = %d, want 1 (no regeneration)", n)
	}
}
