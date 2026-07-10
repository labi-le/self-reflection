package runner

import (
	"context"
	"errors"
	"sync"
	"time"

	"tg2llm/internal/bot"
)

// fakeTransport is a scripted ChatTransport: Updates replays a fixed slice of
// updates and then closes the channel (or, when block is set, holds it open
// until ctx ends), and Send records every delivered message. It makes no
// network call, so the whole run loop is exercised in-process.
type fakeTransport struct {
	updates []bot.Update
	updErr  error // returned by Updates when non-nil (transport failed to start)
	block   bool  // hold the channel open after the scripted updates until ctx ends

	mu   sync.Mutex
	sent []bot.OutgoingMessage
}

func (f *fakeTransport) Updates(ctx context.Context) (<-chan bot.Update, error) {
	if f.updErr != nil {
		return nil, f.updErr
	}
	ch := make(chan bot.Update)
	go func() {
		defer close(ch)
		for _, u := range f.updates {
			select {
			case ch <- u:
			case <-ctx.Done():
				return
			}
		}
		if f.block {
			<-ctx.Done() // keep the channel open so Run must observe cancellation
		}
	}()
	return ch, nil
}

func (f *fakeTransport) Send(_ context.Context, chatID int64, text string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.sent = append(f.sent, bot.OutgoingMessage{ChatID: chatID, Text: text})
	return nil
}

// sends returns a copy of every message delivered so far.
func (f *fakeTransport) sends() []bot.OutgoingMessage {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]bot.OutgoingMessage(nil), f.sent...)
}

// stubBrain is a canned Brain: it records each convo it is asked about and
// returns a fixed reply, a fixed error, or (when delay is set) blocks until the
// bounded context expires — the seam used to exercise the reply-timeout path.
type stubBrain struct {
	reply string
	err   error
	delay time.Duration

	mu    sync.Mutex
	calls [][]bot.Message
}

func (s *stubBrain) Reply(ctx context.Context, convo []bot.Message) (string, error) {
	s.mu.Lock()
	s.calls = append(s.calls, append([]bot.Message(nil), convo...))
	s.mu.Unlock()
	if s.delay > 0 {
		select {
		case <-time.After(s.delay):
		case <-ctx.Done():
			return "", ctx.Err()
		}
	}
	if s.err != nil {
		return "", s.err
	}
	return s.reply, nil
}

// lastConvo is the conversation passed to the most recent Reply call.
func (s *stubBrain) lastConvo() []bot.Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.calls) == 0 {
		return nil
	}
	return s.calls[len(s.calls)-1]
}

func (s *stubBrain) callCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.calls)
}

// errThenOKBrain fails its first Reply and succeeds on every call after, proving
// the loop survives a failed turn and keeps processing later updates.
type errThenOKBrain struct {
	reply string

	mu sync.Mutex
	n  int
}

func (b *errThenOKBrain) Reply(_ context.Context, _ []bot.Message) (string, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.n++
	if b.n == 1 {
		return "", errors.New("transient brain failure")
	}
	return b.reply, nil
}
