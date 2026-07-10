package botapi

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
)

var testLogger = zerolog.Nop()

func allowAll(int64) bool { return true }

// immediateSleep is the delay seam swapped in for tests: it records nothing and
// returns at once, so backoff and inter-segment pacing cost no wall-clock time
// while still honouring cancellation.
func immediateSleep(ctx context.Context, _ time.Duration) bool { return ctx.Err() == nil }

// fakeTG is a scriptable stand-in for the Telegram Bot API. Each getUpdates call
// (after failN leading 500s) returns the next scripted body, then empty results;
// it records the offsets it was polled with and the sendMessage/sendChatAction
// payloads it received, all under a mutex so tests read them race-free.
type fakeTG struct {
	mu          sync.Mutex
	getBodies   []string
	failN       int
	getCalls    int
	sentTexts   []string
	actionCalls int
	offsetCh    chan int64
}

func (f *fakeTG) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	switch {
	case strings.HasSuffix(r.URL.Path, "/getUpdates"):
		f.onGetUpdates(w, body)
	case strings.HasSuffix(r.URL.Path, "/sendMessage"):
		f.onSendMessage(w, body)
	case strings.HasSuffix(r.URL.Path, "/sendChatAction"):
		f.mu.Lock()
		f.actionCalls++
		f.mu.Unlock()
		_, _ = io.WriteString(w, `{"ok":true}`)
	default:
		http.Error(w, "unexpected path", http.StatusNotFound)
	}
}

func (f *fakeTG) onGetUpdates(w http.ResponseWriter, body []byte) {
	var req struct {
		Offset int64 `json:"offset"`
	}
	_ = json.Unmarshal(body, &req)

	f.mu.Lock()
	n := f.getCalls
	f.getCalls++
	f.mu.Unlock()

	if f.offsetCh != nil {
		select {
		case f.offsetCh <- req.Offset:
		default:
		}
	}
	if n < f.failN {
		w.WriteHeader(http.StatusInternalServerError)
		return
	}
	idx := n - f.failN
	f.mu.Lock()
	out := `{"ok":true,"result":[]}`
	if idx < len(f.getBodies) {
		out = f.getBodies[idx]
	}
	f.mu.Unlock()
	_, _ = io.WriteString(w, out)
}

func (f *fakeTG) onSendMessage(w http.ResponseWriter, body []byte) {
	var req struct {
		Text string `json:"text"`
	}
	_ = json.Unmarshal(body, &req)
	f.mu.Lock()
	f.sentTexts = append(f.sentTexts, req.Text)
	f.mu.Unlock()
	_, _ = io.WriteString(w, `{"ok":true}`)
}

func (f *fakeTG) texts() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.sentTexts...)
}

func (f *fakeTG) actions() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.actionCalls
}

func (f *fakeTG) calls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.getCalls
}

// recvUpdate reads one update, failing the test if the channel closes first or
// nothing arrives within the bound.
func recvUpdate(t *testing.T, ch <-chan bot.Update) bot.Update {
	t.Helper()
	select {
	case u, ok := <-ch:
		if !ok {
			t.Fatal("channel closed before an update arrived")
		}
		return u
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for an update")
		return bot.Update{}
	}
}

// waitForOffset drains observed poll offsets until want is seen, or fails.
func waitForOffset(t *testing.T, ch <-chan int64, want int64) {
	t.Helper()
	deadline := time.After(2 * time.Second)
	for {
		select {
		case off := <-ch:
			if off == want {
				return
			}
		case <-deadline:
			t.Fatalf("offset %d never observed", want)
			return
		}
	}
}

// waitForAnyOffset blocks until the loop has polled at least once.
func waitForAnyOffset(t *testing.T, ch <-chan int64) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(2 * time.Second):
		t.Fatal("loop never polled getUpdates")
	}
}

func equalStrings(a, b []string) bool {
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
