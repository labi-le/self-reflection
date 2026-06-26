package media

import (
	"io"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/domain/dialogue"
)

var testLogger = zerolog.New(io.Discard)

// writeFile creates a file with some bytes under root and returns the relative path.
func writeFile(t *testing.T, root, rel string) string {
	t.Helper()
	abs := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(abs), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(abs, []byte("fake-bytes"), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	return rel
}

func TestDecoderPhotoAndVoice(t *testing.T) {
	root := t.TempDir()
	photoRel := writeFile(t, root, "photos/p.jpg")
	voiceRel := writeFile(t, root, "voice_messages/a.ogg")

	photoCalls, voiceCalls := 0, 0
	describe := func(string) string { photoCalls++; return "a cat on a sofa" }
	transcribe := func(string) string { voiceCalls++; return "привет мир" }

	cache := NewCache(filepath.Join(root, "cache.db"), testLogger)
	d := NewDecoder(root, transcribe, describe, cache, map[string]string{"photo": "v1", "voice": "v1"}, testLogger)

	photoMsg := dialogue.Message{HasPhoto: true, PhotoPath: photoRel}
	voiceMsg := dialogue.Message{MediaType: "voice_message", FilePath: voiceRel}

	if got := d.Decode(photoMsg); got != "a cat on a sofa" {
		t.Fatalf("photo decode = %q", got)
	}
	if got := d.Decode(voiceMsg); got != "привет мир" {
		t.Fatalf("voice decode = %q", got)
	}
	if photoCalls != 1 || voiceCalls != 1 {
		t.Fatalf("calls photo=%d voice=%d, want 1/1", photoCalls, voiceCalls)
	}

	// Cache hit: second decode must not call the backend again.
	if got := d.Decode(photoMsg); got != "a cat on a sofa" {
		t.Fatalf("photo decode (cached) = %q", got)
	}
	if photoCalls != 1 {
		t.Fatalf("photo backend called %d times, want 1 (cache hit expected)", photoCalls)
	}
}

func TestDecoderMissingFile(t *testing.T) {
	root := t.TempDir()
	calls := 0
	describe := func(string) string { calls++; return "x" }
	d := NewDecoder(root, nil, describe, nil, nil, testLogger)

	msg := dialogue.Message{HasPhoto: true, PhotoPath: "photos/does-not-exist.jpg"}
	if got := d.Decode(msg); got != "" {
		t.Fatalf("missing file decode = %q, want empty", got)
	}
	if calls != 0 {
		t.Fatalf("backend called %d times for missing file, want 0", calls)
	}
}

func TestDecoderNotIncludedPlaceholder(t *testing.T) {
	root := t.TempDir()
	calls := 0
	describe := func(string) string { calls++; return "x" }
	d := NewDecoder(root, nil, describe, nil, nil, testLogger)

	msg := dialogue.Message{HasPhoto: true, PhotoPath: "(File not included. Change export settings.)"}
	if got := d.Decode(msg); got != "" {
		t.Fatalf("NOT_INCLUDED decode = %q, want empty", got)
	}
	if calls != 0 {
		t.Fatalf("backend called %d times for NOT_INCLUDED, want 0", calls)
	}
}

func TestDecoderEmptyResult(t *testing.T) {
	root := t.TempDir()
	rel := writeFile(t, root, "photos/p.jpg")
	describe := func(string) string { return "   " } // whitespace -> trimmed to empty
	d := NewDecoder(root, nil, describe, NewCache(filepath.Join(root, "c.db"), testLogger), nil, testLogger)

	msg := dialogue.Message{HasPhoto: true, PhotoPath: rel}
	if got := d.Decode(msg); got != "" {
		t.Fatalf("empty-result decode = %q, want empty", got)
	}
}

func TestDecoderVersionSalting(t *testing.T) {
	root := t.TempDir()
	rel := writeFile(t, root, "photos/p.jpg")
	cachePath := filepath.Join(root, "cache.db")

	calls := 0
	describe := func(string) string { calls++; return "desc" }
	msg := dialogue.Message{HasPhoto: true, PhotoPath: rel}

	// Decoder A with photo version "modelA".
	dA := NewDecoder(root, nil, describe, NewCache(cachePath, testLogger), map[string]string{"photo": "modelA"}, testLogger)
	dA.Decode(msg)
	if calls != 1 {
		t.Fatalf("after A: calls=%d, want 1", calls)
	}
	dA.Flush()

	// Decoder B shares the cache file but uses a different photo version -> different key -> re-decode.
	dB := NewDecoder(root, nil, describe, NewCache(cachePath, testLogger), map[string]string{"photo": "modelB"}, testLogger)
	dB.Decode(msg)
	if calls != 2 {
		t.Fatalf("after B (different version): calls=%d, want 2 (no stale hit)", calls)
	}
	dB.Flush()

	// Decoder C reuses modelA version -> should hit the cache A wrote.
	dC := NewDecoder(root, nil, describe, NewCache(cachePath, testLogger), map[string]string{"photo": "modelA"}, testLogger)
	dC.Decode(msg)
	if calls != 2 {
		t.Fatalf("after C (version modelA): calls=%d, want 2 (cache hit expected)", calls)
	}
	dC.Flush()
}

func TestCacheRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache.db")
	c := NewCache(path, testLogger)
	if c == nil {
		t.Fatal("NewCache returned nil")
	}
	c.Put("photo:abc123:v1", "описание на русском")
	if err := c.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	c2 := NewCache(path, testLogger)
	if c2 == nil {
		t.Fatal("reopen returned nil")
	}
	defer c2.Close()
	if got := c2.Get("photo:abc123:v1"); got != "описание на русском" {
		t.Fatalf("reloaded value = %q", got)
	}
	if got := c2.Get("missing"); got != "" {
		t.Fatalf("missing key = %q, want empty", got)
	}
}

func TestNilCacheSafe(t *testing.T) {
	var c *Cache
	if got := c.Get("k"); got != "" {
		t.Fatalf("nil cache Get = %q", got)
	}
	c.Put("k", "v") // must not panic
	if err := c.Close(); err != nil {
		t.Fatalf("nil cache Close err: %v", err)
	}
}

func TestDecoderDedupByContentHash(t *testing.T) {
	root := t.TempDir()
	relA := writeFile(t, root, "photos/a.jpg")
	relB := writeFile(t, root, "photos/b.jpg")

	calls := 0
	describe := func(string) string { calls++; return "same image" }
	d := NewDecoder(root, nil, describe, NewCache(filepath.Join(root, "c.db"), testLogger), map[string]string{"photo": "v1"}, testLogger)

	if got := d.Decode(dialogue.Message{HasPhoto: true, PhotoPath: relA}); got != "same image" {
		t.Fatalf("decode A = %q", got)
	}
	if got := d.Decode(dialogue.Message{HasPhoto: true, PhotoPath: relB}); got != "same image" {
		t.Fatalf("decode B = %q", got)
	}
	if calls != 1 {
		t.Fatalf("identical content decoded %d times, want 1 (content-hash dedup)", calls)
	}
}

// TestDecoderConcurrentSingleflight verifies that many concurrent Decode calls for
// the same content hash collapse onto a single backend invocation (in-flight dedup)
// and every caller observes the same result. Run with -race.
func TestDecoderConcurrentSingleflight(t *testing.T) {
	root := t.TempDir()
	rel := writeFile(t, root, "photos/p.jpg")

	var calls int32
	release := make(chan struct{})
	describe := func(string) string {
		atomic.AddInt32(&calls, 1)
		<-release // hold the in-flight slot so concurrent callers must collapse onto it
		return "one description"
	}
	d := NewDecoder(root, nil, describe, NewCache(filepath.Join(root, "c.db"), testLogger), map[string]string{"photo": "v1"}, testLogger)

	const n = 24
	msg := dialogue.Message{HasPhoto: true, PhotoPath: rel}
	results := make([]string, n)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			<-start
			results[idx] = d.Decode(msg)
		}(i)
	}
	close(start)
	time.Sleep(50 * time.Millisecond) // let followers reach the in-flight wait
	close(release)
	wg.Wait()

	if calls != 1 {
		t.Fatalf("backend called %d times under concurrency, want 1 (singleflight)", calls)
	}
	for i, got := range results {
		if got != "one description" {
			t.Fatalf("result[%d] = %q, want %q", i, got, "one description")
		}
	}
	if err := d.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
}
