package kvcache

import (
	"os"
	"path/filepath"
	"strings"
)

// CacheDirEnv overrides the shared cache directory (used by the Docker images,
// where there is no checked-out module to anchor on).
const CacheDirEnv = "TG2LLM_CACHE_DIR"

// modulePath is the go.mod module this project's checkout declares.
const modulePath = "tg2llm"

// SharedDir is the directory holding the cache databases every command shares:
// $TG2LLM_CACHE_DIR, else <checkout>/cache, else <user cache dir>/tg2llm, else a
// temp directory. Anchoring on the checkout gives every command and every export
// one cache; the later steps keep a binary run from elsewhere from scattering a
// database into whatever directory the user happens to be in.
func SharedDir() string {
	if dir := os.Getenv(CacheDirEnv); dir != "" {
		return dir
	}
	if root, ok := checkoutRoot(); ok {
		return filepath.Join(root, "cache")
	}
	if userCache, err := os.UserCacheDir(); err == nil {
		return filepath.Join(userCache, "tg2llm")
	}
	return filepath.Join(os.TempDir(), "tg2llm")
}

// SharedPath returns the path of cache database name inside SharedDir. It only
// computes a path; New creates the directory when the cache is actually opened.
func SharedPath(name string) string {
	return filepath.Join(SharedDir(), name)
}

// checkoutRoot returns the nearest ancestor of the working directory whose go.mod
// declares this module. It matches the module line rather than accepting any
// go.mod so a binary run from inside an unrelated Go project never drops a
// database of decoded private media into that project.
func checkoutRoot() (string, bool) {
	dir, err := os.Getwd()
	if err != nil {
		return "", false
	}
	for {
		if declaresModule(filepath.Join(dir, "go.mod")) {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

func declaresModule(goMod string) bool {
	b, err := os.ReadFile(goMod)
	if err != nil {
		return false
	}
	for line := range strings.Lines(string(b)) {
		fields := strings.Fields(line)
		if len(fields) >= 2 && fields[0] == "module" && fields[1] == modulePath {
			return true
		}
	}
	return false
}
