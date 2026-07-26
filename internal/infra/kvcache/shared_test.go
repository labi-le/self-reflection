package kvcache

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSharedDirEnvOverride(t *testing.T) {
	t.Setenv(CacheDirEnv, "/var/tmp/tg2llm-cache")
	if got := SharedDir(); got != "/var/tmp/tg2llm-cache" {
		t.Fatalf("SharedDir() = %q, want the env override", got)
	}
	if got := SharedPath("media.db"); got != "/var/tmp/tg2llm-cache/media.db" {
		t.Fatalf("SharedPath() = %q", got)
	}
}

// TestSharedDirFindsCheckoutRoot proves a command run from a subdirectory of the
// checkout still resolves the one cache directory at the module root.
func TestSharedDirFindsCheckoutRoot(t *testing.T) {
	t.Setenv(CacheDirEnv, "")
	root := realPath(t, t.TempDir())
	writeGoMod(t, root, "module tg2llm\n\ngo 1.26\n")
	sub := filepath.Join(root, "cmd", "parse")
	if err := os.MkdirAll(sub, 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	t.Chdir(sub)

	if got, want := SharedDir(), filepath.Join(root, "cache"); got != want {
		t.Fatalf("SharedDir() = %q, want %q", got, want)
	}
}

// TestSharedDirIgnoresForeignModule proves the walk matches this module's go.mod
// only: run from inside someone else's Go project, the cache must not land in it.
func TestSharedDirIgnoresForeignModule(t *testing.T) {
	t.Setenv(CacheDirEnv, "")
	t.Setenv("XDG_CACHE_HOME", realPath(t, t.TempDir()))
	foreign := realPath(t, t.TempDir())
	writeGoMod(t, foreign, "module github.com/someone/else\n")
	t.Chdir(foreign)

	if got := SharedDir(); got == filepath.Join(foreign, "cache") {
		t.Fatalf("SharedDir() = %q, want anywhere but a foreign checkout", got)
	}
}

// TestSharedDirWithoutModule proves the fallback outside a checkout: the per-user
// cache directory, never a cache/ dropped in whatever directory the user stands in.
func TestSharedDirWithoutModule(t *testing.T) {
	t.Setenv(CacheDirEnv, "")
	t.Setenv("XDG_CACHE_HOME", realPath(t, t.TempDir()))
	wd := realPath(t, t.TempDir())
	t.Chdir(wd)
	if _, ok := checkoutRoot(); ok {
		t.Skip("TMPDIR lives inside a tg2llm checkout; the user-cache fallback is unreachable here")
	}
	userCache, err := os.UserCacheDir()
	if err != nil {
		t.Fatalf("UserCacheDir: %v", err)
	}
	if got, want := SharedDir(), filepath.Join(userCache, "tg2llm"); got != want {
		t.Fatalf("SharedDir() = %q, want %q", got, want)
	}
}

// TestNewCreatesParentDir proves opening a cache in a not-yet-existing directory
// (the usual first run against <project>/cache) creates it instead of failing.
func TestNewCreatesParentDir(t *testing.T) {
	path := filepath.Join(t.TempDir(), "cache", "media.db")
	c, err := New(path, "media_cache")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer c.Close()
	if err := c.Put("k", "v"); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("cache file: %v", err)
	}
}

// realPath resolves symlinks (macOS temp dirs) so the expectation matches the
// working directory the OS reports back.
func realPath(t *testing.T, path string) string {
	t.Helper()
	resolved, err := filepath.EvalSymlinks(path)
	if err != nil {
		t.Fatalf("EvalSymlinks: %v", err)
	}
	return resolved
}

func writeGoMod(t *testing.T, dir, content string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(content), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
}
