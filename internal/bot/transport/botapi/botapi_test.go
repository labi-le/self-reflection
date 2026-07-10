package botapi

import (
	"context"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"tg2llm/internal/bot"
)

func Test_Updates_emits_decoded_allowed_update(t *testing.T) {
	// Given a server returning one message from an allowed user
	f := &fakeTG{getBodies: []string{
		`{"ok":true,"result":[{"update_id":42,"message":{"chat":{"id":100},"from":{"id":7},"text":"hi"}}]}`,
	}}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New("secret-token", allowAll, testLogger, WithBaseURL(srv.URL), withSleep(immediateSleep))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// When
	ch, err := c.Updates(ctx)
	if err != nil {
		t.Fatalf("Updates: %v", err)
	}

	// Then the update decodes and is delivered on the channel
	got := recvUpdate(t, ch)
	want := bot.Update{UpdateID: 42, ChatID: 100, UserID: 7, Text: "hi"}
	if got != want {
		t.Fatalf("update = %+v, want %+v", got, want)
	}
}

func Test_Updates_advances_offset_past_delivered_update(t *testing.T) {
	// Given
	f := &fakeTG{
		getBodies: []string{
			`{"ok":true,"result":[{"update_id":42,"message":{"chat":{"id":100},"from":{"id":7},"text":"hi"}}]}`,
		},
		offsetCh: make(chan int64, 16),
	}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New("secret-token", allowAll, testLogger, WithBaseURL(srv.URL), withSleep(immediateSleep))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// When the first update is consumed and the loop polls again
	ch, _ := c.Updates(ctx)
	_ = recvUpdate(t, ch)

	// Then a subsequent poll carries offset = update_id + 1
	waitForOffset(t, f.offsetCh, 43)
}

func Test_Updates_drops_update_from_disallowed_user(t *testing.T) {
	// Given a blocked sender's update first, then an allowed sender's
	f := &fakeTG{getBodies: []string{
		`{"ok":true,"result":[{"update_id":1,"message":{"chat":{"id":100},"from":{"id":999},"text":"blocked"}}]}`,
		`{"ok":true,"result":[{"update_id":2,"message":{"chat":{"id":100},"from":{"id":7},"text":"allowed"}}]}`,
	}}
	srv := httptest.NewServer(f)
	defer srv.Close()
	allowOnly7 := func(id int64) bool { return id == 7 }
	c := New("secret-token", allowOnly7, testLogger, WithBaseURL(srv.URL), withSleep(immediateSleep))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// When
	ch, _ := c.Updates(ctx)

	// Then the blocked update never surfaces: the first delivered update is the
	// allowed id-2 one, proving id-1 was dropped at the edge.
	got := recvUpdate(t, ch)
	if got.UpdateID != 2 || got.UserID != 7 || got.Text != "allowed" {
		t.Fatalf("first delivered update = %+v, want the allowed id-2 update", got)
	}
}

func Test_Send_splits_burst_into_one_message_per_segment(t *testing.T) {
	// Given a delay recorder standing in for the pacing seam
	f := &fakeTG{}
	srv := httptest.NewServer(f)
	defer srv.Close()
	var mu sync.Mutex
	var slept []time.Duration
	rec := func(ctx context.Context, d time.Duration) bool {
		mu.Lock()
		slept = append(slept, d)
		mu.Unlock()
		return ctx.Err() == nil
	}
	c := New("secret-token", allowAll, testLogger, WithBaseURL(srv.URL), withSleep(rec))

	// When three segments joined by the burst delimiter U+2063 are sent
	if err := c.Send(context.Background(), 100, "a\u2063b\u2063c"); err != nil {
		t.Fatalf("Send: %v", err)
	}

	// Then exactly three sendMessage requests, one per segment, in order
	if got, want := f.texts(), []string{"a", "b", "c"}; !equalStrings(got, want) {
		t.Fatalf("sendMessage texts = %q, want %q", got, want)
	}
	// And a human-like pace delay preceded every segment, including the first —
	// the bot no longer answers instantly (3 delays for 3 segments)
	mu.Lock()
	defer mu.Unlock()
	if len(slept) != 3 {
		t.Fatalf("pace delays = %d, want 3", len(slept))
	}
	for i, d := range slept {
		if d <= 0 {
			t.Fatalf("delay %d = %v, want > 0", i, d)
		}
	}
	// And a typing indicator preceded each segment
	if got := f.actions(); got != 3 {
		t.Fatalf("sendChatAction calls = %d, want 3", got)
	}
}

func Test_Updates_retries_after_transient_server_error(t *testing.T) {
	// Given the first poll fails 5xx, the retry returns a real update
	f := &fakeTG{
		failN: 1,
		getBodies: []string{
			`{"ok":true,"result":[{"update_id":7,"message":{"chat":{"id":100},"from":{"id":7},"text":"after retry"}}]}`,
		},
	}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New("secret-token", allowAll, testLogger, WithBaseURL(srv.URL), withSleep(immediateSleep))
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// When
	ch, _ := c.Updates(ctx)

	// Then the 500 is not fatal: the loop retries and eventually delivers
	got := recvUpdate(t, ch)
	if got.UpdateID != 7 || got.Text != "after retry" {
		t.Fatalf("update = %+v, want the post-retry id-7 update", got)
	}
	if got := f.calls(); got < 2 {
		t.Fatalf("getUpdates calls = %d, want >= 2 (a retry happened)", got)
	}
}

func Test_Updates_closes_channel_when_context_cancelled(t *testing.T) {
	// Given a server that always returns empty results
	f := &fakeTG{offsetCh: make(chan int64, 16)}
	srv := httptest.NewServer(f)
	defer srv.Close()
	c := New("secret-token", allowAll, testLogger, WithBaseURL(srv.URL), withSleep(immediateSleep))
	ctx, cancel := context.WithCancel(context.Background())

	// When the loop is running, then the context is cancelled
	ch, _ := c.Updates(ctx)
	waitForAnyOffset(t, f.offsetCh)
	cancel()

	// Then the update channel closes promptly and cleanly
	select {
	case u, ok := <-ch:
		if ok {
			t.Fatalf("received update %+v after cancel, want closed channel", u)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("channel did not close within 2s of cancel")
	}
}

func Test_splitBurst_drops_empty_segments(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{"single segment", "hello", []string{"hello"}},
		{"three segments", "a\u2063b\u2063c", []string{"a", "b", "c"}},
		{"drops empty middle", "a\u2063\u2063b", []string{"a", "b"}},
		{"trims blank-only", "  \u2063x\u2063   ", []string{"x"}},
		{"all empty", "\u2063\u2063", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := splitBurst(tt.in); !equalStrings(got, tt.want) {
				t.Fatalf("splitBurst(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

// randFloat is pinned to 0.5 so the ±jitter factor cancels to 1.0 and the pacing
// arithmetic is exact: a base delay even for an empty reply, growing with length,
// and capped for a very long one.
func Test_typingDelay_paces_like_a_human(t *testing.T) {
	c := New("t", allowAll, testLogger)
	c.randFloat = func() float64 { return 0.5 }

	if got := c.typingDelay(""); got != typingBase {
		t.Errorf("typingDelay(empty) = %v, want base %v", got, typingBase)
	}
	if short, long := c.typingDelay("хай"), c.typingDelay("привет как дела друг"); long <= short {
		t.Errorf("longer message must pace longer: short=%v long=%v", short, long)
	}
	if got := c.typingDelay(strings.Repeat("я", 1000)); got != typingCap {
		t.Errorf("typingDelay(1000 runes) = %v, want cap %v", got, typingCap)
	}
}
