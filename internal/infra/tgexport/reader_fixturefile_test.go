package tgexport

import (
	"io"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"
)

// TestReadMessagesFullAccountFixtureFile drives the reader over a committed,
// representative full-account export fixture (testdata/full_account_mini.json)
// instead of inline JSON. It locks the assertions the inline table tests do not
// make: HasPhoto is set purely by photo-field PRESENCE — true for exactly the one
// photo message (which also carries empty text) and false for every other — over
// a corpus assembled from BOTH chats.list and left_chats.list.
func TestReadMessagesFullAccountFixtureFile(t *testing.T) {
	path := filepath.Join("testdata", "full_account_mini.json")
	r := NewReader(path, zerolog.New(io.Discard))
	msgs, refs, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages(%s): %v", path, err)
	}
	if refs == nil {
		t.Errorf("referencedIDs is nil")
	}

	// The corpus excludes the single service record (id 11); every other typed
	// message with a resolvable date is kept — including the date_unixtime one.
	if len(msgs) != 6 {
		t.Fatalf("len(msgs) = %d, want 6 (service id 11 excluded)", len(msgs))
	}
	byID := indexByID(msgs)

	// Owner is auto-detected from personal_information.user_id, as user<id>.
	if r.OwnerFromID() != "user777001" {
		t.Errorf("OwnerFromID() = %q, want user777001 (from personal_information.user_id)", r.OwnerFromID())
	}

	// HasPhoto is PRESENCE-based: true for exactly the one message carrying a
	// photo field (id 3), false for all others. This "false otherwise" half is
	// what the inline single-chat test never asserts.
	for _, m := range msgs {
		wantPhoto := m.ID == 3
		if m.HasPhoto != wantPhoto {
			t.Errorf("msg %d: HasPhoto = %v, want %v", m.ID, m.HasPhoto, wantPhoto)
		}
	}
	if got := byID[3]; got.PhotoPath == "" || got.ExtractedText != "" {
		t.Errorf("msg 3 = {PhotoPath:%q Text:%q}, want a photo path with empty text", got.PhotoPath, got.ExtractedText)
	}

	// Per-message tagging is drawn from both lists.
	if got := byID[1]; got.FromID != "user200200" || got.ChatType != "personal_chat" || got.ChatID != "200200" {
		t.Errorf("msg 1 = {FromID:%q ChatType:%q ChatID:%q}, want user200200/personal_chat/200200", got.FromID, got.ChatType, got.ChatID)
	}
	if got := byID[10]; got.FromID != "user444555" || got.ChatType != "public_supergroup" || got.ChatID != "300300" {
		t.Errorf("msg 10 = {FromID:%q ChatType:%q ChatID:%q}, want user444555/public_supergroup/300300", got.FromID, got.ChatType, got.ChatID)
	}
	// msg 20 comes from left_chats.list — proving that list is walked too — and
	// its >2^32 from_id tail is preserved verbatim as a string (no truncation).
	if got, ok := byID[20]; !ok || got.FromID != "user5000000001" || got.ChatType != "private_group" || got.ChatID != "500500" {
		t.Errorf("msg 20 = {FromID:%q ChatType:%q ChatID:%q} (ok=%v), want user5000000001/private_group/500500", got.FromID, got.ChatType, got.ChatID, ok)
	}

	// The empty-date / valid-date_unixtime message is retained, not dropped.
	if got, ok := byID[12]; !ok || got.ExtractedText != "unixtime survivor" {
		t.Errorf("msg 12 (date_unixtime fallback) = %+v (ok=%v), want retained with text %q", got, ok, "unixtime survivor")
	}
}
