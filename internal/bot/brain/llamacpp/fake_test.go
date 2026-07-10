package llamacpp

import (
	"encoding/json"
	"io"
	"net/http"
	"sync"
	"testing"
	"time"
)

// fakePersona is a scriptable stand-in for the llama.cpp OpenAI-compatible
// /v1/chat/completions endpoint. It records every request body it receives and
// returns a scripted assistant reply — optionally after a delay, with a non-200
// status, or a verbatim raw body — so tests can drive the Brain without a real
// model. All fields are read under a mutex so httptest's goroutine and the test
// goroutine stay race-free.
type fakePersona struct {
	mu        sync.Mutex
	reply     string        // content returned in choices[0].message.content
	replies   []string      // if set, returned per successive call (last repeats); overrides reply
	status    int           // response status (0 => 200)
	delay     time.Duration // sleep before responding (for timeout tests)
	raw       string        // if set, written verbatim (for malformed/empty-choice bodies)
	gotBodies []string      // raw request bodies received, in order
}

func (f *fakePersona) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	f.mu.Lock()
	f.gotBodies = append(f.gotBodies, string(body))
	reply := f.reply
	if n := len(f.replies); n > 0 {
		if idx := len(f.gotBodies) - 1; idx < n {
			reply = f.replies[idx]
		} else {
			reply = f.replies[n-1]
		}
	}
	status, delay, raw := f.status, f.delay, f.raw
	f.mu.Unlock()

	if delay > 0 {
		select {
		case <-time.After(delay):
		case <-r.Context().Done():
			return // client hung up (ctx cancel / timeout): send nothing
		}
	}
	if status == 0 {
		status = http.StatusOK
	}
	if status != http.StatusOK {
		w.WriteHeader(status)
		return
	}
	if raw != "" {
		_, _ = io.WriteString(w, raw)
		return
	}
	// json.Marshal escapes quotes/Cyrillic/control chars correctly.
	content, _ := json.Marshal(reply)
	_, _ = io.WriteString(w, `{"choices":[{"message":{"content":`+string(content)+`}}]}`)
}

// sentMessage / sentRequest mirror the outbound OpenAI request body so a test can
// decode exactly what the Brain put on the wire (roles, contents, order).
type sentMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type sentRequest struct {
	Model            string        `json:"model"`
	Stream           bool          `json:"stream"`
	Temperature      float64       `json:"temperature"`
	RepeatPenalty    float64       `json:"repeat_penalty"`
	PresencePenalty  float64       `json:"presence_penalty"`
	FrequencyPenalty float64       `json:"frequency_penalty"`
	RepeatLastN      int           `json:"repeat_last_n"`
	Messages         []sentMessage `json:"messages"`
}

// lastRequest decodes the most recent request body the fake received.
func (f *fakePersona) lastRequest(t *testing.T) sentRequest {
	t.Helper()
	f.mu.Lock()
	defer f.mu.Unlock()
	if len(f.gotBodies) == 0 {
		t.Fatal("no request reached the fake persona server")
	}
	var req sentRequest
	if err := json.Unmarshal([]byte(f.gotBodies[len(f.gotBodies)-1]), &req); err != nil {
		t.Fatalf("decode sent request: %v", err)
	}
	return req
}

// callCount returns how many requests the fake has served.
func (f *fakePersona) callCount() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.gotBodies)
}
