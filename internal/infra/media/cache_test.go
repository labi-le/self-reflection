package media

import (
	"path/filepath"
	"testing"
)

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
