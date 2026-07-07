package app_test

import (
	"testing"

	"tg2llm/internal/app"
	"tg2llm/internal/domain/dialogue"
)

// recordingSink captures the messages that reach WriteMessage so a test can
// assert exactly which survive the pipeline's filters (independent of any
// downstream sink formatting).
type recordingSink struct {
	got []dialogue.Message
}

func (r *recordingSink) WriteMessage(msg dialogue.Message, _ string) error {
	r.got = append(r.got, msg)
	return nil
}
func (r *recordingSink) WriteDayHeader(string) error { return nil }
func (r *recordingSink) Close() error                { return nil }

func idsReaching(t *testing.T, opts app.ParseOptions, msgs []dialogue.Message) []int {
	t.Helper()
	rec := &recordingSink{}
	if err := app.NewParseService(fakeSource{msgs: msgs}, rec, nil, opts).Run(); err != nil {
		t.Fatalf("run: %v", err)
	}
	ids := make([]int, 0, len(rec.got))
	for _, m := range rec.got {
		ids = append(ids, m.ID)
	}
	return ids
}

// scopeMsgs is one message per relevant ChatType, each with non-empty text so
// none is dropped by the useless rule.
func scopeMsgs() []dialogue.Message {
	return []dialogue.Message{
		{ID: 1, Date: ts(10, 0, 0), Sender: "A", ChatType: "personal_chat", ChatID: "100", FromID: "user1", ExtractedText: "p"},
		{ID: 2, Date: ts(10, 1, 0), Sender: "B", ChatType: "public_supergroup", ChatID: "200", FromID: "user2", ExtractedText: "g"},
		{ID: 3, Date: ts(10, 2, 0), Sender: "C", ChatType: "private_group", ChatID: "300", FromID: "user3", ExtractedText: "pg"},
		{ID: 4, Date: ts(10, 3, 0), Sender: "D", ChatType: "saved_messages", ChatID: "400", FromID: "user4", ExtractedText: "s"},
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// (chatml) --chat-scope personal keeps ONLY personal_chat messages.
func TestServiceChatScopePersonalChatML(t *testing.T) {
	got := idsReaching(t, app.ParseOptions{Format: "chatml", ChatScope: "personal"}, scopeMsgs())
	want := []int{1}
	if !equalInts(got, want) {
		t.Fatalf("chatml personal scope: ids reaching sink = %v, want %v (only personal_chat)", got, want)
	}
}

// (chatml) --chat-scope all keeps EVERY ChatType. saved_messages is excluded by
// the chatmlsink sink later, NOT by this service-level scope filter, so it still
// reaches the sink here.
func TestServiceChatScopeAllChatML(t *testing.T) {
	got := idsReaching(t, app.ParseOptions{Format: "chatml", ChatScope: "all"}, scopeMsgs())
	want := []int{1, 2, 3, 4}
	if !equalInts(got, want) {
		t.Fatalf("chatml all scope: ids reaching sink = %v, want %v (every type)", got, want)
	}
}

// (text) The scope filter is chatml-only. Default text mode must be unaffected
// even though ChatScope defaults to "personal": every message still reaches the
// sink, preserving byte-identical text output.
func TestServiceChatScopeTextModeUnfiltered(t *testing.T) {
	got := idsReaching(t, app.ParseOptions{Format: "text", ChatScope: "personal"}, scopeMsgs())
	want := []int{1, 2, 3, 4}
	if !equalInts(got, want) {
		t.Fatalf("text mode must ignore --chat-scope: ids reaching sink = %v, want %v", got, want)
	}
}
