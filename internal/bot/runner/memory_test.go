package runner

import (
	"context"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
)

// fakeBuilder records the chats its Maybe was fired for, signalling done so the
// out-of-band (goroutine) trigger can be awaited without flake.
type fakeBuilder struct {
	done chan int64

	mu    sync.Mutex
	chats []int64
}

func (f *fakeBuilder) Maybe(_ context.Context, chatID int64) {
	f.mu.Lock()
	f.chats = append(f.chats, chatID)
	f.mu.Unlock()
	if f.done != nil {
		f.done <- chatID
	}
}

// Given a stored profile, When an allowed sender messages, Then the brain's convo
// begins with a system block carrying the profile, ahead of the user turn.
func Test_handleUpdate_injects_the_stored_profile(t *testing.T) {
	s := allowStore(t)
	if err := s.SaveProfile(testChat, "— зовут Вадик\n— программист", 5); err != nil {
		t.Fatalf("SaveProfile: %v", err)
	}
	br := &stubBrain{reply: "ок"}
	deps := Deps{Transport: &fakeTransport{}, Brain: br, Store: s, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "как меня зовут?"})

	got := br.lastConvo()
	if len(got) < 2 || got[0].Role != roleSystem || !strings.Contains(got[0].Content, "зовут Вадик") {
		t.Fatalf("convo[0] = %+v, want a leading system profile block", got)
	}
	last := got[len(got)-1]
	if last.Role != roleUser || last.Content != "как меня зовут?" {
		t.Fatalf("last convo turn = %+v, want the user turn", last)
	}
}

// Given a builder, When a message is handled, Then it is fired for that chat
// (out-of-band).
func Test_handleUpdate_fires_the_profile_builder(t *testing.T) {
	s := allowStore(t)
	fb := &fakeBuilder{done: make(chan int64, 1)}
	deps := Deps{Transport: &fakeTransport{}, Brain: &stubBrain{reply: "ок"}, Store: s, Builder: fb, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "привет"})

	select {
	case got := <-fb.done:
		if got != testChat {
			t.Fatalf("builder fired for chat %d, want %d", got, testChat)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("profile builder was not fired")
	}
}

// Given no stored profile, When a message is handled, Then no system block is
// injected — the brain sees only the user turn.
func Test_handleUpdate_no_profile_no_block(t *testing.T) {
	s := allowStore(t)
	br := &stubBrain{reply: "ок"}
	deps := Deps{Transport: &fakeTransport{}, Brain: br, Store: s, Logger: zerolog.Nop()}

	handleUpdate(context.Background(), deps, bot.Update{ChatID: testChat, UserID: allowedUID, Text: "привет"})

	got := br.lastConvo()
	if len(got) != 1 || got[0].Role != roleUser {
		t.Fatalf("convo = %+v, want just the user turn (no profile block)", got)
	}
}
