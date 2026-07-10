package runner

import (
	"context"
	"errors"
	"path/filepath"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
	"tg2llm/internal/bot/store"
)

const (
	testOwner  = int64(1)
	testChat   = int64(100)
	allowedUID = int64(7)
	deniedUID  = int64(9)
)

func newStore(t *testing.T) *store.Store {
	t.Helper()
	s := store.New(filepath.Join(t.TempDir(), "state.db"), testOwner, zerolog.Nop())
	if s == nil {
		t.Fatal("store.New returned nil for a valid temp path")
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

func allowStore(t *testing.T) *store.Store {
	t.Helper()
	s := newStore(t)
	if err := s.AddAllowed(allowedUID, "test"); err != nil {
		t.Fatalf("AddAllowed: %v", err)
	}
	return s
}

func assertHistory(t *testing.T, s *store.Store, want []bot.Message) {
	t.Helper()
	got, err := s.LoadRecent(testChat, 100)
	if err != nil {
		t.Fatalf("LoadRecent: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("history = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("history[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// Given an allowed sender and empty history, When handleUpdate runs, Then the
// brain sees the inbound turn, both turns are persisted oldest-first, and the
// reply is delivered to the sender's chat.
func Test_handleUpdate_allowed_persists_both_turns_and_sends(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{reply: "здарова"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "привет"})

	got := br.lastConvo()
	want := []bot.Message{{Role: "user", Content: "привет"}}
	if len(got) != len(want) || got[0] != want[0] {
		t.Fatalf("brain convo = %+v, want %+v", got, want)
	}
	assertHistory(t, s, []bot.Message{
		{Role: "user", Content: "привет"},
		{Role: "assistant", Content: "здарова"},
	})
	sent := tr.sends()
	if len(sent) != 1 || sent[0].ChatID != testChat || sent[0].Text != "здарова" {
		t.Fatalf("sent = %+v, want one message %q to chat %d", sent, "здарова", testChat)
	}
}

// Given prior history for the chat, When an allowed sender messages, Then the
// brain receives that history followed by the new inbound turn (context loaded).
func Test_handleUpdate_loads_recent_context_before_the_new_turn(t *testing.T) {
	s := allowStore(t)
	if err := s.AppendMessage(testChat, "user", "как дела", 1); err != nil {
		t.Fatalf("seed AppendMessage: %v", err)
	}
	if err := s.AppendMessage(testChat, "assistant", "норм", 2); err != nil {
		t.Fatalf("seed AppendMessage: %v", err)
	}
	br := &stubBrain{reply: "ага"}
	deps := Deps{Transport: &fakeTransport{}, Brain: br, Store: s, Logger: zerolog.Nop(), ContextTurns: 10}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "что нового"})

	got := br.lastConvo()
	want := []bot.Message{
		{Role: "user", Content: "как дела"},
		{Role: "assistant", Content: "норм"},
		{Role: "user", Content: "что нового"},
	}
	if len(got) != len(want) {
		t.Fatalf("brain convo = %+v, want %+v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("brain convo[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// Given a sender absent from the allowlist, When handleUpdate runs, Then the
// update is dropped: the brain is never consulted, nothing is persisted, and
// nothing is sent.
func Test_handleUpdate_denied_sender_is_dropped(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{reply: "should not happen"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: deniedUID, Text: "let me in"})

	if br.callCount() != 0 {
		t.Fatalf("brain called %d times for a denied sender, want 0", br.callCount())
	}
	assertHistory(t, s, nil)
	if sent := tr.sends(); len(sent) != 0 {
		t.Fatalf("sent = %+v, want nothing sent for a denied sender", sent)
	}
}

// Given the configured owner (never inserted into allowed_ids), When they
// message, Then the update is handled — the owner is always allowed.
func Test_handleUpdate_owner_is_always_allowed(t *testing.T) {
	s := newStore(t) // note: owner is NOT added to allowed_ids
	tr := &fakeTransport{}
	br := &stubBrain{reply: "да"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: testOwner, Text: "я дома"})

	if sent := tr.sends(); len(sent) != 1 || sent[0].Text != "да" {
		t.Fatalf("sent = %+v, want the owner's message answered", sent)
	}
}

// Given draft mode, When an allowed sender messages, Then the reply is generated
// and both turns persisted, but nothing is sent (generate-but-don't-deliver).
func Test_handleUpdate_draft_generates_and_persists_but_does_not_send(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{reply: "черновик"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop(), Draft: true}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "привет"})

	if br.callCount() != 1 {
		t.Fatalf("brain called %d times, want 1 (draft still generates)", br.callCount())
	}
	assertHistory(t, s, []bot.Message{
		{Role: "user", Content: "привет"},
		{Role: "assistant", Content: "черновик"},
	})
	if sent := tr.sends(); len(sent) != 0 {
		t.Fatalf("sent = %+v, want nothing sent in draft mode", sent)
	}
}

// Given the send kill-switch, When an allowed sender messages, Then the reply is
// generated and persisted but never delivered.
func Test_handleUpdate_send_disabled_suppresses_delivery(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{reply: "тихо"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop(), SendDisabled: true}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "эй"})

	if br.callCount() != 1 {
		t.Fatalf("brain called %d times, want 1", br.callCount())
	}
	if sent := tr.sends(); len(sent) != 0 {
		t.Fatalf("sent = %+v, want nothing sent when send is disabled", sent)
	}
}

// Given a brain that errors, When handleUpdate runs for an allowed sender, Then
// it does not panic and delivers nothing; the inbound turn is still persisted
// (step c) but no assistant turn is written (step e is skipped).
func Test_handleUpdate_brain_error_is_handled_gracefully(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{err: errors.New("boom")}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "привет"})

	if sent := tr.sends(); len(sent) != 0 {
		t.Fatalf("sent = %+v, want nothing sent when the brain errors", sent)
	}
	assertHistory(t, s, []bot.Message{{Role: "user", Content: "привет"}})
}

// Given a brain slower than the reply timeout, When handleUpdate runs, Then the
// bounded context expires, the error is swallowed, and nothing is sent — proving
// the Brain call is timeout-bounded.
func Test_handleUpdate_brain_timeout_is_bounded_and_swallowed(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{reply: "too late", delay: time.Second}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop(), ReplyTimeout: 10 * time.Millisecond}

	done := make(chan struct{})
	go func() {
		handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "привет"})
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("handleUpdate did not return; reply timeout not enforced")
	}
	if sent := tr.sends(); len(sent) != 0 {
		t.Fatalf("sent = %+v, want nothing sent on brain timeout", sent)
	}
}

// Given a scripted transport delivering a turn whose reply fails then a good
// turn, When Run drives the loop, Then only the good reply is delivered — one
// failed turn never stops the loop — and Run returns nil when the channel closes.
func Test_Run_continues_after_a_failed_turn_and_stops_on_channel_close(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{updates: []bot.Update{
		{ChatID: testChat, UserID: allowedUID, Text: "first"},
		{ChatID: testChat, UserID: allowedUID, Text: "second"},
	}}
	br := &errThenOKBrain{reply: "ok"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := Run(ctx, deps); err != nil {
		t.Fatalf("Run: %v", err)
	}
	sent := tr.sends()
	if len(sent) != 1 || sent[0].Text != "ok" {
		t.Fatalf("sent = %+v, want exactly the second (good) reply delivered", sent)
	}
}

// Given a transport whose Updates fails to start, When Run is called, Then it
// returns that error rather than silently spinning.
func Test_Run_returns_transport_startup_error(t *testing.T) {
	tr := &fakeTransport{updErr: errors.New("cannot start")}
	deps := Deps{Transport: tr, Brain: &stubBrain{}, Store: newStore(t), Logger: zerolog.Nop()}

	if err := Run(context.Background(), deps); err == nil {
		t.Fatal("Run returned nil, want the transport startup error")
	}
}

// Given a transport that holds its channel open, When the context is cancelled,
// Then Run returns nil promptly (graceful shutdown, no hang).
func Test_Run_stops_on_context_cancel(t *testing.T) {
	tr := &fakeTransport{block: true}
	deps := Deps{Transport: tr, Brain: &stubBrain{}, Store: newStore(t), Logger: zerolog.Nop()}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- Run(ctx, deps) }()
	cancel()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not stop on context cancel")
	}
}
