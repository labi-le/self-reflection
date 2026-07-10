package llamacpp

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"tg2llm/internal/bot"
)

// lockedPersonaSystem is the byte-identical persona system prompt the fine-tune
// was trained with. It is duplicated here on purpose as a golden guard: if the
// package constant ever drifts, this test fails and flags a persona/contract
// break (the fine-tuned model expects this exact string).
const lockedPersonaSystem = "Ты — Денис. Отвечай в своём обычном стиле, как в личной переписке."

func Test_defaultPersonaSystem_is_byte_identical_to_the_locked_prompt(t *testing.T) {
	if bot.DefaultPersonaSystem != lockedPersonaSystem {
		t.Fatalf("persona system drifted:\n got  = %q\n want = %q", bot.DefaultPersonaSystem, lockedPersonaSystem)
	}
}

// Blank persona falls back to the shared constant — the serve-side half of the
// train↔serve parity that cmd/parse's --persona-system default anchors.
func Test_New_blank_persona_uses_shared_default(t *testing.T) {
	c := New("http://persona.invalid", "", "")
	if c.persona != bot.DefaultPersonaSystem {
		t.Fatalf("blank-persona New used %q, want shared default %q", c.persona, bot.DefaultPersonaSystem)
	}
}

// Test (a): the outgoing request injects the persona as messages[0] (byte-identical)
// followed by the conversation mapped 1:1 in order.
func Test_Reply_sends_persona_system_first_then_conversation_in_order(t *testing.T) {
	// Given a persona server that records what it receives
	f := &fakePersona{reply: "ок"}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "", WithTimeout(2*time.Second))
	convo := []bot.Message{
		{Role: "user", Content: "привет"},
		{Role: "assistant", Content: "здарова"},
		{Role: "user", Content: "как ты?"},
	}

	// When
	if _, err := c.Reply(context.Background(), convo); err != nil {
		t.Fatalf("Reply: %v", err)
	}

	// Then messages[0] is the locked persona system turn, byte-identical...
	got := f.lastRequest(t)
	want := []sentMessage{
		{Role: "system", Content: lockedPersonaSystem},
		{Role: "user", Content: "привет"},
		{Role: "assistant", Content: "здарова"},
		{Role: "user", Content: "как ты?"},
	}
	if len(got.Messages) != len(want) {
		t.Fatalf("messages = %d (%+v), want %d", len(got.Messages), got.Messages, len(want))
	}
	for i := range want {
		if got.Messages[i] != want[i] {
			t.Fatalf("messages[%d] = %+v, want %+v", i, got.Messages[i], want[i])
		}
	}
	// ...and the request is a non-streaming completion at the default temperature.
	if got.Stream {
		t.Fatal("stream = true, want false")
	}
	if got.Temperature != defaultTemperature {
		t.Fatalf("temperature = %v, want %v", got.Temperature, defaultTemperature)
	}
}

// Test: a caller-supplied persona string is what gets sent (proves it is
// config-driven, not hardcoded to the default).
func Test_Reply_uses_the_configured_persona_string(t *testing.T) {
	f := &fakePersona{reply: "ок"}
	srv := httptest.NewServer(f)
	defer srv.Close()
	custom := "Ты — другой персонаж."
	c := New(srv.URL, custom, "")

	if _, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "hi"}}); err != nil {
		t.Fatalf("Reply: %v", err)
	}
	got := f.lastRequest(t)
	if got.Messages[0].Role != "system" || got.Messages[0].Content != custom {
		t.Fatalf("messages[0] = %+v, want system/%q", got.Messages[0], custom)
	}
}

// Test (b): Reply returns the mock's choices[0].message.content, trimmed.
func Test_Reply_returns_trimmed_assistant_content(t *testing.T) {
	f := &fakePersona{reply: "  привет, как дела?\n"}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "")

	got, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "?"}})
	if err != nil {
		t.Fatalf("Reply: %v", err)
	}
	if want := "привет, как дела?"; got != want {
		t.Fatalf("Reply = %q, want %q (trimmed)", got, want)
	}
}

// Test (e): an ordinary reply is returned unchanged.
func Test_Reply_passes_benign_reply_through_unchanged(t *testing.T) {
	benign := "ага, давай завтра погуляем в парке"
	f := &fakePersona{reply: benign}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "")

	got, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "гуляем?"}})
	if err != nil {
		t.Fatalf("Reply: %v", err)
	}
	if got != benign {
		t.Fatalf("Reply = %q, want the benign reply unchanged %q", got, benign)
	}
}

// Test (c): a slow server plus a bounded context makes Reply return an error
// promptly instead of hanging.
func Test_Reply_returns_error_on_context_timeout_without_hanging(t *testing.T) {
	f := &fakePersona{reply: "too late", delay: 5 * time.Second}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "", WithTimeout(80*time.Millisecond))

	done := make(chan error, 1)
	go func() {
		_, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "hi"}})
		done <- err
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Reply returned nil error on a timed-out request, want an error")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Reply hung past the request timeout instead of returning an error")
	}
}

// Test: an explicit ctx cancellation is honoured too.
func Test_Reply_returns_error_when_ctx_cancelled(t *testing.T) {
	f := &fakePersona{reply: "unused", delay: 5 * time.Second}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "", WithTimeout(10*time.Second))
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan error, 1)
	go func() {
		_, err := c.Reply(ctx, []bot.Message{{Role: "user", Content: "hi"}})
		done <- err
	}()
	cancel()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Reply returned nil error after ctx cancel, want an error")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("Reply hung after ctx cancel")
	}
}

// Test: a non-200 status is a clear error, not a silent empty reply.
func Test_Reply_errors_on_non_200(t *testing.T) {
	f := &fakePersona{status: 500}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "")

	if _, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "hi"}}); err == nil {
		t.Fatal("Reply returned nil error on HTTP 500, want an error")
	}
}

// Test: an empty choices array is an error (no reply to return).
func Test_Reply_errors_on_empty_choices(t *testing.T) {
	f := &fakePersona{raw: `{"choices":[]}`}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "")

	if _, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "hi"}}); err == nil {
		t.Fatal("Reply returned nil error on empty choices, want an error")
	}
}

// Test: a malformed JSON body is a decode error.
func Test_Reply_errors_on_malformed_body(t *testing.T) {
	f := &fakePersona{raw: `not json at all`}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New(srv.URL, "", "")

	if _, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "hi"}}); err == nil {
		t.Fatal("Reply returned nil error on malformed body, want an error")
	}
}
