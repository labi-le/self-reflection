package kvcache

import (
	"path/filepath"
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
