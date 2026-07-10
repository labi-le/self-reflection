package main

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot/brain/llamacpp"
	"tg2llm/internal/bot/runner"
	"tg2llm/internal/bot/store"
	"tg2llm/internal/bot/transport/botapi"
)

// TestBotEndToEndAgainstMocks drives the REAL bot wiring — the Telegram Bot API
// transport, the llama.cpp persona Brain, the SQLite store and the runner loop —
// against two httptest servers standing in for Telegram and the persona model
// server. It proves the whole ingress -> allowlist gate -> brain -> persist ->
// egress path works end-to-end with mocks, so swapping the mock persona server
// for the real fine-tuned model is a one-flag change (--llama-host). It also
// exercises the allowlist gate: a disallowed sender in the same batch is
// silently dropped and never answered.
func TestBotEndToEndAgainstMocks(t *testing.T) {
	const (
		ownerID      = int64(1)
		allowedUser  = int64(4242)
		allowedChat  = int64(4242)
		blockedUser  = int64(999)
		blockedChat  = int64(999)
		userText     = "привет, как дела?"
		personaReply = "привет! у меня всё хорошо, рад тебя слышать"
	)

	// --- persona model mock: OpenAI-compatible /v1/chat/completions ---
	var (
		pmu      sync.Mutex
		seenMsgs []wireMsg
	)
	persona := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.Error(w, "unexpected path", http.StatusNotFound)
			return
		}
		var req struct {
			Messages []wireMsg `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		pmu.Lock()
		seenMsgs = req.Messages
		pmu.Unlock()
		_ = json.NewEncoder(w).Encode(cannedCompletion(personaReply))
	}))
	defer persona.Close()

	// --- Telegram Bot API mock: one batch with a blocked + an allowed sender ---
	batch, err := json.Marshal(map[string]any{
		"ok": true,
		"result": []map[string]any{
			{"update_id": 1, "message": map[string]any{
				"chat": map[string]any{"id": blockedChat},
				"from": map[string]any{"id": blockedUser},
				"text": "я не в списке",
			}},
			{"update_id": 2, "message": map[string]any{
				"chat": map[string]any{"id": allowedChat},
				"from": map[string]any{"id": allowedUser},
				"text": userText,
			}},
		},
	})
	if err != nil {
		t.Fatalf("marshal getUpdates batch: %v", err)
	}
	tg := &tgMock{batch: string(batch), delivered: make(chan sentMsg, 4)}
	tgServer := httptest.NewServer(tg)
	defer tgServer.Close()

	// --- real state, transport, brain, runner (exactly what cmd/bot wires) ---
	st := store.New(filepath.Join(t.TempDir(), "state.db"), ownerID, zerolog.Nop())
	if st == nil {
		t.Fatal("store.New returned nil")
	}
	defer st.Close()
	if err := st.AddAllowed(allowedUser, "integration-test"); err != nil {
		t.Fatalf("seed allowlist: %v", err)
	}

	transport := botapi.New("TESTTOKEN", st.IsAllowed, zerolog.Nop(), botapi.WithBaseURL(tgServer.URL))
	brain := llamacpp.New(persona.URL, "", "", llamacpp.WithTimeout(5*time.Second))

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	done := make(chan error, 1)
	go func() {
		done <- runner.Run(ctx, runner.Deps{
			Transport:    transport,
			Brain:        brain,
			Store:        st,
			Logger:       zerolog.Nop(),
			ContextTurns: 10,
			ReplyTimeout: 5 * time.Second,
		})
	}()

	// --- the allowed user's message is answered with the persona reply ---
	var got sentMsg
	select {
	case got = <-tg.delivered:
	case <-time.After(5 * time.Second):
		t.Fatal("no reply delivered within 5s")
	}
	if got.Text != personaReply {
		t.Errorf("delivered text = %q, want %q", got.Text, personaReply)
	}
	if got.ChatID != allowedChat {
		t.Errorf("delivered to chat %d, want %d", got.ChatID, allowedChat)
	}

	// --- the disallowed sender is dropped at the gate: no second delivery ---
	select {
	case extra := <-tg.delivered:
		t.Errorf("unexpected extra delivery (blocked sender should be silent): %+v", extra)
	case <-time.After(300 * time.Millisecond):
	}

	// --- the brain received the locked persona system turn + the user turn ---
	pmu.Lock()
	msgs := seenMsgs
	pmu.Unlock()
	if len(msgs) < 2 {
		t.Fatalf("persona saw %d messages, want >= 2", len(msgs))
	}
	if msgs[0].Role != "system" || !strings.Contains(msgs[0].Content, "Денис") {
		t.Errorf("first turn = %+v, want a persona system turn mentioning Денис", msgs[0])
	}
	if last := msgs[len(msgs)-1]; last.Role != "user" || last.Content != userText {
		t.Errorf("last turn = %+v, want user %q", last, userText)
	}

	// --- both turns were persisted for the allowed chat ---
	convo, err := st.LoadRecent(allowedChat, 10)
	if err != nil {
		t.Fatalf("load recent: %v", err)
	}
	if len(convo) != 2 ||
		convo[0].Role != "user" || convo[0].Content != userText ||
		convo[1].Role != "assistant" || convo[1].Content != personaReply {
		t.Errorf("persisted conversation = %+v, want [user:%q, assistant:%q]", convo, userText, personaReply)
	}

	// --- clean shutdown: cancel stops the loop promptly ---
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("runner did not stop after context cancel")
	}
}

// wireMsg is one chat turn as seen on the OpenAI-compatible wire.
type wireMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// cannedCompletion builds a minimal /v1/chat/completions response whose first
// choice carries reply.
func cannedCompletion(reply string) map[string]any {
	return map[string]any{
		"choices": []map[string]any{
			{"message": map[string]any{"role": "assistant", "content": reply}},
		},
	}
}

// sentMsg is a message the bot delivered to the Telegram mock.
type sentMsg struct {
	ChatID int64
	Text   string
}

// tgMock is a scriptable stand-in for the Telegram Bot API: the first getUpdates
// call returns batch, later calls return an empty result; sendMessage payloads
// are surfaced on delivered; sendChatAction is accepted. It mirrors the wire
// contract in internal/bot/transport/botapi.
type tgMock struct {
	mu        sync.Mutex
	callN     int
	batch     string
	delivered chan sentMsg
}

func (m *tgMock) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	switch {
	case strings.HasSuffix(r.URL.Path, "/getUpdates"):
		m.mu.Lock()
		n := m.callN
		m.callN++
		m.mu.Unlock()
		if n == 0 {
			_, _ = io.WriteString(w, m.batch)
			return
		}
		time.Sleep(20 * time.Millisecond) // gentle long-poll stand-in
		_, _ = io.WriteString(w, `{"ok":true,"result":[]}`)
	case strings.HasSuffix(r.URL.Path, "/sendMessage"):
		var req struct {
			ChatID int64  `json:"chat_id"`
			Text   string `json:"text"`
		}
		_ = json.Unmarshal(body, &req)
		select {
		case m.delivered <- sentMsg{ChatID: req.ChatID, Text: req.Text}:
		default:
		}
		_, _ = io.WriteString(w, `{"ok":true}`)
	case strings.HasSuffix(r.URL.Path, "/sendChatAction"):
		_, _ = io.WriteString(w, `{"ok":true}`)
	default:
		http.Error(w, "unexpected path", http.StatusNotFound)
	}
}
