package store

import (
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
)

var testLogger = zerolog.Nop()

const ownerID = int64(42)

func newTemp(t *testing.T) *Store {
	t.Helper()
	s := New(filepath.Join(t.TempDir(), "state.db"), ownerID, testLogger)
	if s == nil {
		t.Fatal("New returned nil for a valid temp path")
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// Given a fresh store, When the owner id is checked, Then it is allowed even
// though it was never inserted into allowed_ids; an unrelated id is rejected.
func TestOwnerAlwaysAllowed(t *testing.T) {
	s := newTemp(t)
	if !s.IsAllowed(ownerID) {
		t.Fatal("owner must be allowed even when absent from allowed_ids")
	}
	if s.IsAllowed(999) {
		t.Fatal("an unlisted, non-owner id must be rejected")
	}
}

// Given a store constructed with ownerID=0 (unset), When any id (including 0)
// is checked, Then the owner short-circuit is NOT fail-open: only ids present
// in allowed_ids are permitted.
func TestUnsetOwnerIsNotFailOpen(t *testing.T) {
	s := New(filepath.Join(t.TempDir(), "state.db"), 0, testLogger)
	if s == nil {
		t.Fatal("New returned nil for a valid temp path")
	}
	t.Cleanup(func() { _ = s.Close() })

	if s.IsAllowed(0) {
		t.Fatal("ownerID=0 must not fail-open for userID=0")
	}
	if s.IsAllowed(123) {
		t.Fatal("ownerID=0 must not allow arbitrary ids")
	}
	if err := s.AddAllowed(7, "friend"); err != nil {
		t.Fatalf("AddAllowed: %v", err)
	}
	if !s.IsAllowed(7) {
		t.Fatal("a seeded allowed id must still return true when ownerID=0")
	}
}

// Given a store with a configured owner, When userID=0 is checked, Then the
// owner short-circuit does NOT accidentally match the zero-valued unset id.
func TestConfiguredOwnerDoesNotAllowZero(t *testing.T) {
	s := newTemp(t)
	if !s.IsAllowed(ownerID) {
		t.Fatal("configured owner must be allowed")
	}
	if s.IsAllowed(0) {
		t.Fatal("userID=0 must not match ownerID=42")
	}
}

// Given a fresh store, When an id is added then removed, Then IsAllowed tracks
// the transition true -> ... -> false.
func TestAddThenRemoveAllowed(t *testing.T) {
	s := newTemp(t)
	const uid = int64(111)

	if s.IsAllowed(uid) {
		t.Fatal("precondition: 111 must not be allowed before AddAllowed")
	}
	if err := s.AddAllowed(uid, "friend"); err != nil {
		t.Fatalf("AddAllowed: %v", err)
	}
	if !s.IsAllowed(uid) {
		t.Fatal("after AddAllowed(111), IsAllowed(111) must be true")
	}
	if err := s.RemoveAllowed(uid); err != nil {
		t.Fatalf("RemoveAllowed: %v", err)
	}
	if s.IsAllowed(uid) {
		t.Fatal("after RemoveAllowed(111), IsAllowed(111) must be false")
	}
}

// Given several appended turns across two chats, When LoadRecent(chat, n) is
// called, Then it returns exactly the most-recent n rows of THAT chat in
// chronological order, ignoring other chats.
func TestAppendLoadRecentRoundTrip(t *testing.T) {
	s := newTemp(t)
	const chat = int64(7)

	turns := []bot.Message{
		{Role: "user", Content: "one"},
		{Role: "assistant", Content: "two"},
		{Role: "user", Content: "three"},
	}
	for i, m := range turns {
		if err := s.AppendMessage(chat, m.Role, m.Content, int64(1000+i)); err != nil {
			t.Fatalf("AppendMessage[%d]: %v", i, err)
		}
	}
	// A different chat must never bleed into chat 7's history.
	if err := s.AppendMessage(99, "user", "other-chat", 1234); err != nil {
		t.Fatalf("AppendMessage(other chat): %v", err)
	}

	got, err := s.LoadRecent(chat, 2)
	if err != nil {
		t.Fatalf("LoadRecent(n=2): %v", err)
	}
	want := []bot.Message{
		{Role: "assistant", Content: "two"},
		{Role: "user", Content: "three"},
	}
	if len(got) != len(want) {
		t.Fatalf("LoadRecent(n=2) returned %d rows (%+v), want %d", len(got), got, len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("LoadRecent(n=2)[%d] = %+v, want %+v", i, got[i], want[i])
		}
	}

	// n greater than the row count returns every turn, still chronological.
	all, err := s.LoadRecent(chat, 10)
	if err != nil {
		t.Fatalf("LoadRecent(n=10): %v", err)
	}
	if len(all) != 3 || all[0].Content != "one" || all[2].Content != "three" {
		t.Fatalf("LoadRecent(n=10) = %+v, want the 3 turns oldest-first", all)
	}
}

// Given history for two chats, When ClearHistory clears chat A, Then chat A's
// history is emptied while chat B's survives — a clean is scoped to one chat and
// never nukes another chat's context.
func TestClearHistoryIsScopedToOneChat(t *testing.T) {
	s := newTemp(t)
	const chatA, chatB = int64(7), int64(8)

	for _, m := range []bot.Message{
		{Role: "user", Content: "a1"},
		{Role: "assistant", Content: "a2"},
	} {
		if err := s.AppendMessage(chatA, m.Role, m.Content, 1); err != nil {
			t.Fatalf("AppendMessage(A): %v", err)
		}
	}
	if err := s.AppendMessage(chatB, "user", "b1", 1); err != nil {
		t.Fatalf("AppendMessage(B): %v", err)
	}

	if err := s.ClearHistory(chatA); err != nil {
		t.Fatalf("ClearHistory(A): %v", err)
	}

	gotA, err := s.LoadRecent(chatA, 100)
	if err != nil {
		t.Fatalf("LoadRecent(A): %v", err)
	}
	if len(gotA) != 0 {
		t.Fatalf("chat A history after ClearHistory = %+v, want empty", gotA)
	}
	gotB, err := s.LoadRecent(chatB, 100)
	if err != nil {
		t.Fatalf("LoadRecent(B): %v", err)
	}
	if len(gotB) != 1 || gotB[0].Content != "b1" {
		t.Fatalf("chat B history = %+v, want it untouched (b1)", gotB)
	}
}

// Given a nil *Store, When any method is called, Then it degrades to a safe
// no-op / default without panicking (fail-closed for the allowlist gate).
func TestNilStoreSafe(t *testing.T) {
	var s *Store
	if s.IsAllowed(ownerID) {
		t.Fatal("nil store must fail closed: IsAllowed == false")
	}
	if err := s.AddAllowed(1, "x"); err != nil {
		t.Fatalf("nil AddAllowed must be a no-op, got %v", err)
	}
	if err := s.RemoveAllowed(1); err != nil {
		t.Fatalf("nil RemoveAllowed must be a no-op, got %v", err)
	}
	if err := s.AppendMessage(1, "user", "hi", 1); err != nil {
		t.Fatalf("nil AppendMessage must be a no-op, got %v", err)
	}
	if err := s.ClearHistory(1); err != nil {
		t.Fatalf("nil ClearHistory must be a no-op, got %v", err)
	}
	got, err := s.LoadRecent(1, 5)
	if err != nil {
		t.Fatalf("nil LoadRecent must be a no-op, got err %v", err)
	}
	if got != nil {
		t.Fatalf("nil LoadRecent must return nil slice, got %+v", got)
	}
	if err := s.Close(); err != nil {
		t.Fatalf("nil Close must be a no-op, got %v", err)
	}
}

// Given an empty path, When New is called, Then it returns nil so the bot runs
// storeless (mirrors cache.go's uncached contract).
func TestEmptyPathReturnsNil(t *testing.T) {
	if s := New("", ownerID, testLogger); s != nil {
		t.Fatal(`New("") must return nil`)
	}
}

// Given a store file written once, When it is re-opened, Then the CREATE ...
// IF NOT EXISTS schema re-runs without error and the prior row survives.
func TestSchemaIdempotentAcrossReopen(t *testing.T) {
	path := filepath.Join(t.TempDir(), "state.db")

	s1 := New(path, ownerID, testLogger)
	if s1 == nil {
		t.Fatal("first New returned nil")
	}
	if err := s1.AddAllowed(5, "seed"); err != nil {
		t.Fatalf("AddAllowed: %v", err)
	}
	if err := s1.Close(); err != nil {
		t.Fatalf("Close(1): %v", err)
	}

	s2 := New(path, ownerID, testLogger)
	if s2 == nil {
		t.Fatal("re-open returned nil (schema not idempotent?)")
	}
	t.Cleanup(func() { _ = s2.Close() })
	if !s2.IsAllowed(5) {
		t.Fatal("re-opened store lost the previously-added allowlist row")
	}
}

func newTempOpen(t *testing.T) *Store {
	t.Helper()
	s := New(filepath.Join(t.TempDir(), "state.db"), ownerID, testLogger, WithOpenWhenEmpty())
	if s == nil {
		t.Fatal("New returned nil for a valid temp path")
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// Given a store built WithOpenWhenEmpty, When the allowlist is empty, Then any
// id is allowed (owner-curated bootstrap); When the first id is added, Then only
// owner + allowed_ids pass; When every id is removed, Then it reopens to all.
func TestOpenWhenEmptyAllowsUntilFirstAdd(t *testing.T) {
	s := newTempOpen(t)

	if !s.IsAllowed(123456) {
		t.Fatal("WithOpenWhenEmpty + empty allowlist must allow an arbitrary id")
	}

	const added = int64(7)
	if err := s.AddAllowed(added, "first"); err != nil {
		t.Fatalf("AddAllowed: %v", err)
	}
	if s.IsAllowed(123456) {
		t.Fatal("after the first add, a non-added, non-owner id must be rejected")
	}
	if !s.IsAllowed(added) {
		t.Fatal("the added id must be allowed")
	}
	if !s.IsAllowed(ownerID) {
		t.Fatal("the owner must always be allowed")
	}

	if err := s.RemoveAllowed(added); err != nil {
		t.Fatalf("RemoveAllowed: %v", err)
	}
	if !s.IsAllowed(123456) {
		t.Fatal("after removing the last id, the empty allowlist must reopen to everyone")
	}
}

// Given a store built WITHOUT the option, When the allowlist is empty, Then a
// non-owner id is still rejected — the default stays fail-closed.
func TestDefaultEmptyAllowlistIsFailClosed(t *testing.T) {
	s := newTemp(t)
	if s.IsAllowed(123456) {
		t.Fatal("without WithOpenWhenEmpty an empty allowlist must reject non-owner ids")
	}
}

// Given stores with and without a configured owner, When IsOwner is checked,
// Then only a configured, matching id is the owner; ownerID=0 and a nil store
// are never owners.
func TestIsOwner(t *testing.T) {
	s := newTemp(t)
	if !s.IsOwner(ownerID) {
		t.Fatal("IsOwner must be true for the configured owner")
	}
	if s.IsOwner(ownerID + 1) {
		t.Fatal("IsOwner must be false for a non-owner id")
	}

	z := New(filepath.Join(t.TempDir(), "state.db"), 0, testLogger)
	if z == nil {
		t.Fatal("New returned nil for a valid temp path")
	}
	t.Cleanup(func() { _ = z.Close() })
	if z.IsOwner(0) {
		t.Fatal("ownerID=0 (unset) must not make userID=0 the owner")
	}

	var nilStore *Store
	if nilStore.IsOwner(ownerID) {
		t.Fatal("a nil store must never report an owner")
	}
}

// Given ids added with and without a note, When ListAllowed is called, Then it
// returns every entry with its note; a nil store yields (nil, nil).
func TestListAllowedReturnsEntriesWithNotes(t *testing.T) {
	s := newTemp(t)

	empty, err := s.ListAllowed()
	if err != nil {
		t.Fatalf("ListAllowed (fresh): %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("ListAllowed on a fresh store = %+v, want empty", empty)
	}

	if err := s.AddAllowed(7, "friend"); err != nil {
		t.Fatalf("AddAllowed(7): %v", err)
	}
	if err := s.AddAllowed(8, ""); err != nil {
		t.Fatalf("AddAllowed(8): %v", err)
	}

	got, err := s.ListAllowed()
	if err != nil {
		t.Fatalf("ListAllowed: %v", err)
	}
	// Order-independent: same-second added_at values make ORDER BY added_at a
	// tie, so assert the set of entries rather than a fixed order.
	byID := make(map[int64]string, len(got))
	for _, e := range got {
		byID[e.UserID] = e.Note
	}
	if len(byID) != 2 || byID[7] != "friend" || byID[8] != "" {
		t.Fatalf("ListAllowed = %+v, want {7:\"friend\", 8:\"\"}", got)
	}

	var nilStore *Store
	nilGot, err := nilStore.ListAllowed()
	if err != nil || nilGot != nil {
		t.Fatalf("nil ListAllowed = (%+v, %v), want (nil, nil)", nilGot, err)
	}
}
