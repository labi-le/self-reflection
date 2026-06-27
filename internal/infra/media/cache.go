package media

import (
	"database/sql"

	"github.com/rs/zerolog"
	"tg2llm/pkg/ctxlog"

	_ "modernc.org/sqlite"
)

// Cache is a SQLite-backed key/value store that persists decoded-media results so
// expensive operations (whisper transcription, vision description) run once per
// file. Keys are derived from the file's content hash (see Decoder.decodePath),
// so identical media is decoded a single time regardless of its path.
type Cache struct {
	db     *sql.DB
	logger zerolog.Logger
}

// NewCache opens (or creates) a SQLite cache at path. Returns nil if path is empty
// or on any setup error, so decoding simply proceeds uncached.
func NewCache(path string, logger zerolog.Logger) *Cache {
	if path == "" {
		return nil
	}
	logger = logger.With().Str("component", "media-cache").Logger()
	ctxLog := ctxlog.Op(logger, "Cache.New").With().Str("path", path).Logger()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("cache open failed")
		return nil
	}
	// SQLite allows a single writer; one connection avoids intra-process
	// SQLITE_BUSY and serializes our sequential writes.
	db.SetMaxOpenConns(1)
	for _, pragma := range []string{
		"PRAGMA journal_mode=WAL;",  // concurrent reader + writer across processes
		"PRAGMA busy_timeout=5000;", // wait up to 5s on a lock instead of erroring
	} {
		if _, err := db.Exec(pragma); err != nil {
			db.Close()
			ctxLog.Warn().Err(err).Str("pragma", pragma).Msg("cache pragma failed")
			return nil
		}
	}
	if _, err := db.Exec(`CREATE TABLE IF NOT EXISTS media_cache (
		key   TEXT PRIMARY KEY,
		value TEXT NOT NULL
	);`); err != nil {
		db.Close()
		ctxLog.Warn().Err(err).Msg("cache schema failed")
		return nil
	}
	return &Cache{db: db, logger: logger}
}

// Get returns the cached value for key, or "" if absent (or on any error).
func (c *Cache) Get(key string) string {
	if c == nil || c.db == nil {
		return ""
	}
	ctxLog := ctxlog.Op(c.logger, "Cache.Get").With().Str("key", key).Logger()
	var value string
	err := c.db.QueryRow("SELECT value FROM media_cache WHERE key = ?", key).Scan(&value)
	if err != nil {
		if err != sql.ErrNoRows {
			ctxLog.Warn().Err(err).Msg("cache get failed")
		}
		return ""
	}
	return value
}

// Put stores key->value (insert or replace), committed immediately. Best-effort:
// errors are logged, never fatal.
func (c *Cache) Put(key, value string) {
	if c == nil || c.db == nil {
		return
	}
	ctxLog := ctxlog.Op(c.logger, "Cache.Put").With().Str("key", key).Logger()
	if _, err := c.db.Exec(
		"INSERT INTO media_cache(key, value) VALUES(?, ?) "+
			"ON CONFLICT(key) DO UPDATE SET value = excluded.value",
		key, value,
	); err != nil {
		ctxLog.Warn().Err(err).Msg("cache put failed")
	}
}

// Close checkpoints the WAL into the main file and closes the database.
// Safe to call on a nil cache.
func (c *Cache) Close() error {
	if c == nil || c.db == nil {
		return nil
	}
	// Fold the WAL back so the cache is a single file at rest.
	c.db.Exec("PRAGMA wal_checkpoint(TRUNCATE);")
	return c.db.Close()
}
