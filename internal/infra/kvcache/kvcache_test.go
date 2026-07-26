package kvcache

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
)

func TestPutGetRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "kv.db")
	c, err := New(path, "custom_table")
	if err != nil {
		t.Fatalf("new: %v", err)
	}
	if err := c.Put("photo:abc123:v1", "описание на русском"); err != nil {
		t.Fatalf("put: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("close: %v", err)
	}

	c2, err := New(path, "custom_table")
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer c2.Close()
	if got, ok := c2.Get("photo:abc123:v1"); !ok || got != "описание на русском" {
		t.Fatalf("Get = (%q, %v), want (%q, true)", got, ok, "описание на русском")
	}
	if got, ok := c2.Get("missing"); ok || got != "" {
		t.Fatalf("missing key = (%q, %v), want (\"\", false)", got, ok)
	}
}

func TestNilCacheSafe(t *testing.T) {
	var c *Cache
	if got, ok := c.Get("k"); ok || got != "" {
		t.Fatalf("nil cache Get = (%q, %v), want (\"\", false)", got, ok)
	}
	if err := c.Put("k", "v"); err != nil {
		t.Fatalf("nil cache Put err: %v", err)
	}
	if err := c.Close(); err != nil {
		t.Fatalf("nil cache Close err: %v", err)
	}
}

func TestInvalidTableName(t *testing.T) {
	path := filepath.Join(t.TempDir(), "kv.db")
	for _, bad := range []string{"", "1abc", "bad name", "drop;table", "a-b", "t;--"} {
		if _, err := New(path, bad); err == nil {
			t.Errorf("New with table %q = nil error, want error", bad)
		}
	}
}

func TestSignature(t *testing.T) {
	a := Signature("qwen2.5vl:7b", "prompt")
	if a != Signature("qwen2.5vl:7b", "prompt") {
		t.Fatal("Signature is not deterministic")
	}
	if len(a) != 10 {
		t.Fatalf("Signature length = %d, want 10", len(a))
	}
	if Signature("modelA", "p") == Signature("modelB", "p") {
		t.Error("distinct models produced identical signatures")
	}
}

// TestNewConcurrentFirstOpens proves that simultaneous first opens of one shared
// cache all succeed and all write. It guards the statement ORDER inside New: with
// journal_mode=WAL issued before busy_timeout, these eight openers fail. The
// cross-process half of the contract needs separate processes — see
// TestNewCrossProcessFirstOpens.
func TestNewConcurrentFirstOpens(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache", "media.db")
	const n = 8

	caches := make([]*Cache, n)
	errs := make([]error, n)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			<-start
			caches[i], errs[i] = New(path, "media_cache")
		}(i)
	}
	close(start)
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("concurrent open %d: %v", i, err)
		}
		if err := caches[i].Put(fmt.Sprintf("k%d", i), "v"); err != nil {
			t.Fatalf("put from opener %d: %v", i, err)
		}
		if err := caches[i].Close(); err != nil {
			t.Fatalf("close %d: %v", i, err)
		}
	}

	c, err := New(path, "media_cache")
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer c.Close()
	for i := range n {
		if got, ok := c.Get(fmt.Sprintf("k%d", i)); !ok || got != "v" {
			t.Errorf("key k%d = (%q, %v), want (\"v\", true)", i, got, ok)
		}
	}
}

// The child re-exec reads its cache path and key from the environment.
const (
	childCachePathEnv = "TG2LLM_TEST_CACHE_PATH"
	childCacheKeyEnv  = "TG2LLM_TEST_CACHE_KEY"
)

// TestNewCrossProcessFirstOpens re-execs this test binary against one fresh cache.
// Only a separate PROCESS can lose the WAL-switch race — an in-process busy
// handler covers it — so this is the test that fails if New is ever "simplified"
// to return the journal_mode=WAL error instead of tolerating it.
func TestNewCrossProcessFirstOpens(t *testing.T) {
	if path := os.Getenv(childCachePathEnv); path != "" {
		c, err := New(path, "media_cache")
		if err != nil {
			t.Fatalf("child open: %v", err)
		}
		if err := c.Put(os.Getenv(childCacheKeyEnv), "v"); err != nil {
			t.Fatalf("child put: %v", err)
		}
		if err := c.Close(); err != nil {
			t.Fatalf("child close: %v", err)
		}
		return
	}

	path := filepath.Join(t.TempDir(), "cache", "media.db")
	const n = 12
	outs := make([]string, n)
	errs := make([]error, n)
	var wg sync.WaitGroup
	for i := range n {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			cmd := exec.Command(os.Args[0], "-test.run=^TestNewCrossProcessFirstOpens$")
			cmd.Env = append(os.Environ(),
				childCachePathEnv+"="+path,
				fmt.Sprintf("%s=k%d", childCacheKeyEnv, i))
			b, err := cmd.CombinedOutput()
			outs[i], errs[i] = string(b), err
		}(i)
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			t.Fatalf("child %d failed to open the shared cache: %v\n%s", i, err, outs[i])
		}
	}

	c, err := New(path, "media_cache")
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer c.Close()
	for i := range n {
		if got, ok := c.Get(fmt.Sprintf("k%d", i)); !ok || got != "v" {
			t.Errorf("key k%d = (%q, %v), want (\"v\", true) — a child's write was lost", i, got, ok)
		}
	}
}
