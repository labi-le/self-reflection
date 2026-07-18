package media

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"

	"tg2llm/internal/infra/kvcache"
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

// newCache opens a media_cache-backed kvcache at path, failing the test on error.
func newCache(t *testing.T, path string) *kvcache.Cache {
	t.Helper()
	c, err := kvcache.New(path, "media_cache")
	if err != nil {
		t.Fatalf("kvcache.New: %v", err)
	}
	return c
}
