package runner

import (
	"context"
	"strconv"
	"strings"
	"testing"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
	"tg2llm/internal/bot/store"
)

// Given assorted texts, When isCommand classifies them, Then only a
// leading-slash (after trimming) counts as a command.
func Test_isCommand(t *testing.T) {
	cases := []struct {
		text string
		want bool
	}{
		{"/whoami", true},
		{"  /help", true},
		{"/allow 5 note", true},
		{"привет", false},
		{"", false},
		{"not /a command", false},
		{"   ", false},
	}
	for _, tc := range cases {
		if got := isCommand(tc.text); got != tc.want {
			t.Errorf("isCommand(%q) = %v, want %v", tc.text, got, tc.want)
		}
	}
}

// Given a command routed straight to handleCommand (past the allow-gate), When
// it dispatches, Then it consumes the update, replies once to the sender's chat,
// never touches the brain, and mutates the allowlist only for an owner.
func Test_handleCommand_dispatch(t *testing.T) {
	cases := []struct {
		name            string
		seedIDs         []int64
		sender          int64
		text            string
		wantReplySubstr string
		check           func(t *testing.T, s *store.Store)
	}{
		{
			name:            "whoami replies the sender id",
			sender:          allowedUID,
			text:            "/whoami",
			wantReplySubstr: "твой id: 7",
		},
		{
			name:            "whoami tolerates a @botname suffix",
			sender:          allowedUID,
			text:            "/whoami@my_persona_bot",
			wantReplySubstr: "твой id: 7",
		},
		{
			name:            "id is an alias of whoami",
			sender:          allowedUID,
			text:            "/id",
			wantReplySubstr: "твой id: 7",
		},
		{
			name:            "owner allow adds with a note and replies",
			sender:          testOwner,
			text:            "/allow 555 best friend",
			wantReplySubstr: "добавил 555",
			check: func(t *testing.T, s *store.Store) {
				if !s.IsAllowed(555) {
					t.Fatal("/allow 555 must add 555 to the allowlist")
				}
			},
		},
		{
			name:            "owner deny removes and replies",
			seedIDs:         []int64{555},
			sender:          testOwner,
			text:            "/deny 555",
			wantReplySubstr: "убрал 555",
			check: func(t *testing.T, s *store.Store) {
				if s.IsAllowed(555) {
					t.Fatal("/deny 555 must remove 555 from the allowlist")
				}
			},
		},
		{
			name:            "owner allow without an id reports the format",
			sender:          testOwner,
			text:            "/allow",
			wantReplySubstr: "формат: /allow",
		},
		{
			name:            "non-owner allow is rejected and adds nobody",
			sender:          allowedUID,
			text:            "/allow 555",
			wantReplySubstr: ownerOnly,
			check: func(t *testing.T, s *store.Store) {
				if s.IsAllowed(555) {
					t.Fatal("a non-owner /allow must not add anyone")
				}
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := allowStore(t)
			for _, id := range tc.seedIDs {
				if err := s.AddAllowed(id, "seed"); err != nil {
					t.Fatalf("seed AddAllowed(%d): %v", id, err)
				}
			}
			tr := &fakeTransport{}
			br := &stubBrain{reply: "unused"}
			deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

			if !handleCommand(context.Background(), deps, bot.Update{ChatID: testChat, UserID: tc.sender, Text: tc.text}) {
				t.Fatalf("handleCommand(%q) = false, want it consumed", tc.text)
			}
			sent := tr.sends()
			if len(sent) != 1 {
				t.Fatalf("sent = %+v, want exactly one reply", sent)
			}
			if sent[0].ChatID != testChat {
				t.Fatalf("reply chat = %d, want %d", sent[0].ChatID, testChat)
			}
			if !strings.Contains(sent[0].Text, tc.wantReplySubstr) {
				t.Fatalf("reply %q does not contain %q", sent[0].Text, tc.wantReplySubstr)
			}
			if br.callCount() != 0 {
				t.Fatalf("brain called %d times for a command, want 0", br.callCount())
			}
			if tc.check != nil {
				tc.check(t, s)
			}
		})
	}
}

// Given several allowed ids (one with an empty note), When the owner runs /list,
// Then each entry is one line with a header count and the dash is omitted for an
// empty note.
func Test_handleCommand_owner_list_formats_entries(t *testing.T) {
	s := allowStore(t)
	if err := s.AddAllowed(555, "bestie"); err != nil {
		t.Fatalf("AddAllowed(555): %v", err)
	}
	if err := s.AddAllowed(556, ""); err != nil {
		t.Fatalf("AddAllowed(556): %v", err)
	}
	tr := &fakeTransport{}
	deps := Deps{Transport: tr, Brain: &stubBrain{}, Store: s, Logger: zerolog.Nop()}

	if !handleCommand(context.Background(), deps, bot.Update{ChatID: testChat, UserID: testOwner, Text: "/list"}) {
		t.Fatal("/list must be consumed")
	}
	sent := tr.sends()
	if len(sent) != 1 {
		t.Fatalf("sent = %+v, want one reply", sent)
	}
	out := sent[0].Text
	for _, want := range []string{"разрешены (3):", "7 — test", "555 — bestie", "556"} {
		if !strings.Contains(out, want) {
			t.Fatalf("/list reply %q is missing %q", out, want)
		}
	}
	if strings.Contains(out, "556 —") {
		t.Fatalf("/list reply %q must omit the dash for an empty note", out)
	}
}

// Given an owner with no allowed ids, When they run /list, Then the reply says
// the list is empty.
func Test_handleCommand_owner_list_empty_says_so(t *testing.T) {
	s := newStore(t)
	tr := &fakeTransport{}
	deps := Deps{Transport: tr, Brain: &stubBrain{}, Store: s, Logger: zerolog.Nop()}

	if !handleCommand(context.Background(), deps, bot.Update{ChatID: testChat, UserID: testOwner, Text: "/list"}) {
		t.Fatal("/list must be consumed")
	}
	sent := tr.sends()
	if len(sent) != 1 || !strings.Contains(sent[0].Text, "список пуст") {
		t.Fatalf("sent = %+v, want 'список пуст'", sent)
	}
}

// Given the same /help, When an owner versus a non-owner asks, Then the owner
// sees the admin commands and the non-owner sees only the /whoami hint.
func Test_handleCommand_help_is_owner_aware(t *testing.T) {
	s := allowStore(t)

	ownerTr := &fakeTransport{}
	ownerDeps := Deps{Transport: ownerTr, Brain: &stubBrain{}, Store: s, Logger: zerolog.Nop()}
	if !handleCommand(context.Background(), ownerDeps, bot.Update{ChatID: testChat, UserID: testOwner, Text: "/help"}) {
		t.Fatal("/help must be consumed")
	}
	ownerSent := ownerTr.sends()
	if len(ownerSent) != 1 || !strings.Contains(ownerSent[0].Text, "/allow") {
		t.Fatalf("owner /help = %+v, want it to list /allow", ownerSent)
	}

	userTr := &fakeTransport{}
	userDeps := Deps{Transport: userTr, Brain: &stubBrain{}, Store: s, Logger: zerolog.Nop()}
	if !handleCommand(context.Background(), userDeps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "/help"}) {
		t.Fatal("/help must be consumed")
	}
	userSent := userTr.sends()
	if len(userSent) != 1 || !strings.Contains(userSent[0].Text, "/whoami") {
		t.Fatalf("non-owner /help = %+v, want the minimal /whoami hint", userSent)
	}
	if strings.Contains(userSent[0].Text, "/allow") {
		t.Fatalf("non-owner /help must not reveal admin commands: %q", userSent[0].Text)
	}
}

// Given an unknown slash-command, When handleCommand runs, Then it is consumed
// silently: no reply, and the persona brain is never asked to answer it.
func Test_handleCommand_unknown_is_consumed_silently(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{reply: "x"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

	if !handleCommand(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "/nonsense"}) {
		t.Fatal("an unknown command must be consumed so the persona never answers it")
	}
	if sent := tr.sends(); len(sent) != 0 {
		t.Fatalf("unknown command must reply nothing, got %+v", sent)
	}
	if br.callCount() != 0 {
		t.Fatalf("unknown command must not reach the brain, got %d calls", br.callCount())
	}
}

// Given seeded conversation history for the sender's chat, When they run /clean,
// Then it is consumed, replies a casual confirmation, and their history is
// cleared — a sender wipes only their own context.
func Test_handleCommand_clean_clears_history(t *testing.T) {
	s := allowStore(t)
	if err := s.AppendMessage(testChat, "user", "привет", 1); err != nil {
		t.Fatalf("seed AppendMessage: %v", err)
	}
	if err := s.AppendMessage(testChat, "assistant", "здарова", 2); err != nil {
		t.Fatalf("seed AppendMessage: %v", err)
	}
	tr := &fakeTransport{}
	deps := Deps{Transport: tr, Brain: &stubBrain{}, Store: s, Logger: zerolog.Nop()}

	if !handleCommand(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "/clean"}) {
		t.Fatal("/clean must be consumed")
	}
	sent := tr.sends()
	if len(sent) != 1 || !strings.Contains(sent[0].Text, "почистил") {
		t.Fatalf("sent = %+v, want a /clean confirmation containing 'почистил'", sent)
	}
	assertHistory(t, s, nil)
}

// Given a command from an allowed sender, When handleUpdate runs, Then the
// command is answered but the brain is never called and nothing is persisted as
// conversation — commands bypass the persona entirely.
func Test_handleUpdate_command_bypasses_the_brain_and_is_not_persisted(t *testing.T) {
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{reply: "should not happen"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: testOwner, Text: "/whoami"})

	if br.callCount() != 0 {
		t.Fatalf("brain called %d times for a command, want 0", br.callCount())
	}
	assertHistory(t, s, nil)
	sent := tr.sends()
	want := "твой id: " + strconv.FormatInt(testOwner, 10)
	if len(sent) != 1 || !strings.Contains(sent[0].Text, want) {
		t.Fatalf("sent = %+v, want the /whoami reply %q", sent, want)
	}
}

// Given an ordinary (non-command) message from an allowed sender, When
// handleUpdate runs, Then it is not treated as a command and reaches the brain.
func Test_handleUpdate_normal_message_reaches_the_brain(t *testing.T) {
	if isCommand("привет") {
		t.Fatal("a normal message must not be classified as a command")
	}
	s := allowStore(t)
	tr := &fakeTransport{}
	br := &stubBrain{reply: "здарова"}
	deps := Deps{Transport: tr, Brain: br, Store: s, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "привет"})

	if br.callCount() != 1 {
		t.Fatalf("brain called %d times for a normal message, want 1", br.callCount())
	}
	sent := tr.sends()
	if len(sent) != 1 || sent[0].Text != "здарова" {
		t.Fatalf("sent = %+v, want the brain's reply delivered", sent)
	}
}
