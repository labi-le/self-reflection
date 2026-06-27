package media

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"
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
