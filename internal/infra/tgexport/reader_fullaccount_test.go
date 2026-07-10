package tgexport

import (
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/domain/dialogue"
)

// indexByID returns a lookup of produced messages keyed by their ID. IDs are not
// globally unique across chats in a real export, but the small fixtures here use
// distinct IDs so the lookup is unambiguous.
func indexByID(msgs []dialogue.Message) map[int]dialogue.Message {
	out := make(map[int]dialogue.Message, len(msgs))
	for _, m := range msgs {
		out[m.ID] = m
	}
	return out
}

// TestReadMessagesFullAccountIteratesBothLists locks the full-account shape:
// chats.list[] AND left_chats.list[] are both walked, every message is tagged
// with its chat type/id, service records are excluded, and the owner is taken
// from personal_information.user_id.
func TestReadMessagesFullAccountIteratesBothLists(t *testing.T) {
	payload := `{
		"personal_information": {"user_id": 100},
		"chats": {"list": [
			{"name":"Alice","type":"personal_chat","id":200,"messages":[
				{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Alice","from_id":"user200","text":"hi"},
				{"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"Me","from_id":"user100","text":"hello"}
			]},
			{"name":"Group","type":"public_supergroup","id":300,"messages":[
				{"id":10,"type":"message","date":"2026-06-19T11:00:00","from":"Bob","from_id":"user400","text":"yo"},
				{"id":11,"type":"service","date":"2026-06-19T11:01:00","actor_id":"user400","action":"pin_message"}
			]}
		]},
		"left_chats": {"list": [
			{"name":"OldChat","type":"private_group","id":500,"messages":[
				{"id":20,"type":"message","date":"2026-06-19T12:00:00","from":"Carol","from_id":"user600","text":"bye"}
			]}
		]}
	}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	msgs, refs, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) != 4 {
		t.Fatalf("len(msgs) = %d, want 4 (service excluded)", len(msgs))
	}
	byID := indexByID(msgs)

	if got := byID[1]; got.ChatType != "personal_chat" || got.ChatID != "200" || got.FromID != "user200" {
		t.Errorf("msg 1 = {ChatType:%q ChatID:%q FromID:%q}, want personal_chat/200/user200", got.ChatType, got.ChatID, got.FromID)
	}
	if got := byID[10]; got.ChatType != "public_supergroup" || got.ChatID != "300" {
		t.Errorf("msg 10 = {ChatType:%q ChatID:%q}, want public_supergroup/300", got.ChatType, got.ChatID)
	}
	// Proves left_chats.list[] was iterated.
	if got, ok := byID[20]; !ok || got.ChatType != "private_group" || got.ChatID != "500" {
		t.Errorf("msg 20 from left_chats = %+v (ok=%v), want private_group/500", got, ok)
	}
	if _, leaked := byID[11]; leaked {
		t.Errorf("service message id 11 leaked into the corpus")
	}
	if refs == nil {
		t.Errorf("referencedIDs is nil")
	}
	if r.OwnerFromID() != "user100" {
		t.Errorf("OwnerFromID() = %q, want user100 (from personal_information.user_id)", r.OwnerFromID())
	}
}

// TestReadMessagesCapturesFromIDAndActorID proves from_id and actor_id are
// captured verbatim onto the produced Message (prefixed strings, not ints).
func TestReadMessagesCapturesFromIDAndActorID(t *testing.T) {
	payload := `{
		"personal_information": {"user_id": 100},
		"chats": {"list": [
			{"name":"c","type":"personal_chat","id":200,"messages":[
				{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"X","from_id":"user200","actor_id":"channel777","text":"hi"}
			]}
		]}
	}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	msgs, _, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want 1", len(msgs))
	}
	if msgs[0].FromID != "user200" {
		t.Errorf("FromID = %q, want user200", msgs[0].FromID)
	}
	if msgs[0].ActorID != "channel777" {
		t.Errorf("ActorID = %q, want channel777", msgs[0].ActorID)
	}
}

// TestReadMessagesDateUnixtimeFallback fixes the latent data-loss bug: a message
// with an empty string date but a valid date_unixtime must be retained, with its
// timestamp derived from the unix seconds.
func TestReadMessagesDateUnixtimeFallback(t *testing.T) {
	payload := `{
		"personal_information": {"user_id": 100},
		"chats": {"list": [
			{"name":"c","type":"personal_chat","id":200,"messages":[
				{"id":1,"type":"message","date":"","date_unixtime":"1700000000","from":"X","from_id":"user200","text":"hi"}
			]}
		]}
	}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	msgs, _, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want 1 (date_unixtime message must be retained)", len(msgs))
	}
	want := time.Unix(1700000000, 0).UTC()
	if !msgs[0].Date.Equal(want) {
		t.Errorf("Date = %v, want %v (from date_unixtime)", msgs[0].Date, want)
	}
}

// TestReadMessagesBothDatesMissingSkipped proves a message with neither a
// parseable date nor a date_unixtime is skipped (only then).
func TestReadMessagesBothDatesMissingSkipped(t *testing.T) {
	payload := `{
		"personal_information": {"user_id": 100},
		"chats": {"list": [
			{"name":"c","type":"personal_chat","id":200,"messages":[
				{"id":1,"type":"message","date":"","from":"X","from_id":"user200","text":"gone"},
				{"id":2,"type":"message","date":"2026-06-19T10:00:00","from":"X","from_id":"user200","text":"kept"}
			]}
		]}
	}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	msgs, _, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) != 1 || msgs[0].ID != 2 {
		t.Fatalf("msgs = %+v, want only id 2 (id 1 has no date and no date_unixtime)", msgs)
	}
}

// TestReadMessagesServiceDroppedButReferenced proves a service record is excluded
// from the corpus yet its reply_to target is still recorded in referencedIDs.
func TestReadMessagesServiceDroppedButReferenced(t *testing.T) {
	payload := `{
		"personal_information": {"user_id": 100},
		"chats": {"list": [
			{"name":"c","type":"personal_chat","id":200,"messages":[
				{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"X","from_id":"user200","text":"target"},
				{"id":99,"type":"service","date":"2026-06-19T10:01:00","actor_id":"user200","action":"phone_call","reply_to_message_id":1}
			]}
		]}
	}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	msgs, refs, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	for _, m := range msgs {
		if m.ID == 99 {
			t.Fatalf("service message id 99 leaked into the corpus")
		}
	}
	if !refs[1] {
		t.Errorf("refs[1] missing: the service message's reply_to target must be recorded, got %v", refs)
	}
}

// TestReadMessagesOwnerHintWins locks priority (1): an explicit hint overrides
// even a present personal_information.user_id.
func TestReadMessagesOwnerHintWins(t *testing.T) {
	payload := `{
		"personal_information": {"user_id": 100},
		"chats": {"list": [
			{"name":"c","type":"personal_chat","id":200,"messages":[
				{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"X","from_id":"user200","text":"hi"}
			]}
		]}
	}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard)).WithOwnerHint("user999")
	if _, _, err := r.ReadMessages(); err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if r.OwnerFromID() != "user999" {
		t.Errorf("OwnerFromID() = %q, want user999 (hint wins over personal_information)", r.OwnerFromID())
	}
}

// TestReadMessagesOwnerSingleChat1on1 locks priority (3): in a single-chat
// personal_chat export, the owner is the user from_id whose numeric tail differs
// from the top-level chat id (the counterparty).
func TestReadMessagesOwnerSingleChat1on1(t *testing.T) {
	payload := `{"name":"Them","type":"personal_chat","id":222,"messages":[
		{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Them","from_id":"user222","text":"hi"},
		{"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"Me","from_id":"user111","text":"hello"}
	]}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	if _, _, err := r.ReadMessages(); err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if r.OwnerFromID() != "user111" {
		t.Errorf("OwnerFromID() = %q, want user111 (tail != chat id 222)", r.OwnerFromID())
	}
}

// TestReadMessagesOwnerAmbiguousFullAccountErrors locks priority (4): a
// full-account export whose owner cannot be resolved (no personal_information,
// no hint) fails loudly and names the --owner-user-id flag.
func TestReadMessagesOwnerAmbiguousFullAccountErrors(t *testing.T) {
	payload := `{
		"chats": {"list": [
			{"name":"Group","type":"private_group","id":500,"messages":[
				{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"A","from_id":"user1","text":"a"},
				{"id":2,"type":"message","date":"2026-06-19T10:01:00","from":"B","from_id":"user2","text":"b"},
				{"id":3,"type":"message","date":"2026-06-19T10:02:00","from":"C","from_id":"user3","text":"c"}
			]}
		]}
	}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	_, _, err := r.ReadMessages()
	if err == nil {
		t.Fatal("ReadMessages: err = nil, want a loud owner-detection error")
	}
	if !strings.Contains(err.Error(), "--owner-user-id") {
		t.Errorf("error %q does not mention --owner-user-id", err.Error())
	}
}

// TestReadMessagesLargeFromIDTailSurvives proves a >32-bit (indeed >53-bit)
// from_id tail is preserved exactly as a string, never truncated through an int.
func TestReadMessagesLargeFromIDTailSurvives(t *testing.T) {
	const bigID = "user123456789012345" // tail ~1.2e17, far beyond int32 and 2^53
	payload := `{
		"personal_information": {"user_id": 100},
		"chats": {"list": [
			{"name":"c","type":"personal_chat","id":200,"messages":[
				{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"X","from_id":"` + bigID + `","text":"hi"}
			]}
		]}
	}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	msgs, _, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) != 1 || msgs[0].FromID != bigID {
		t.Fatalf("FromID = %q, want exact %q", msgs[0].FromID, bigID)
	}
}

// TestReadMessagesSingleChatTaggedFromRoot pins the enriched single-chat path:
// a single-chat object export tags messages with the root type/id and captures
// from_id, while the legacy behaviour (sender, text, reply) is unchanged.
func TestReadMessagesSingleChatTaggedFromRoot(t *testing.T) {
	payload := `{"name":"Them","type":"personal_chat","id":222,"messages":[
		{"id":1,"type":"message","date":"2026-06-19T10:00:00","from":"Them","from_id":"user222","text":"hi"}
	]}`
	r := NewReader(writeJSON(t, payload), zerolog.New(io.Discard))
	msgs, _, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want 1", len(msgs))
	}
	m := msgs[0]
	if m.ChatType != "personal_chat" || m.ChatID != "222" || m.FromID != "user222" {
		t.Errorf("tagging = {ChatType:%q ChatID:%q FromID:%q}, want personal_chat/222/user222", m.ChatType, m.ChatID, m.FromID)
	}
	if m.Sender != "Them" || m.ExtractedText != "hi" {
		t.Errorf("legacy fields = {Sender:%q Text:%q}, want Them/hi", m.Sender, m.ExtractedText)
	}
}

// TestReadMessagesRealDump is an env-guarded probe over the real 141 MB
// full-account export. It never prints message content. Enable with
// TG2LLM_REALDUMP=1.
func TestReadMessagesRealDump(t *testing.T) {
	if os.Getenv("TG2LLM_REALDUMP") != "1" {
		t.Skip("set TG2LLM_REALDUMP=1 to run the real-dump probe")
	}
	const path = "/home/labile/GolandProjects/tg2llm/result.json"
	if _, err := os.Stat(path); err != nil {
		t.Skipf("real dump not available: %v", err)
	}
	r := NewReader(path, zerolog.New(io.Discard))
	msgs, _, err := r.ReadMessages()
	if err != nil {
		t.Fatalf("ReadMessages: %v", err)
	}
	if len(msgs) <= 250000 {
		t.Errorf("message count = %d, want > 250000", len(msgs))
	}
	if r.OwnerFromID() != "user395448554" {
		t.Errorf("OwnerFromID() = %q, want user395448554", r.OwnerFromID())
	}
	t.Logf("real dump: %d messages, owner=%s", len(msgs), r.OwnerFromID())
}
