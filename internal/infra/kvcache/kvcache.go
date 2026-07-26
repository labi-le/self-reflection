// Package kvcache is a content-hash-keyed SQLite key/value store shared by
// callers that memoise expensive results across runs. A single connection
// serialises writes, WAL journalling allows a concurrent reader across
// processes, and all methods are nil-safe so a caller configured without a
// path simply runs uncached.
package kvcache

import (
	"crypto/sha1"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	_ "modernc.org/sqlite"
)

// validTable guards the table identifier: SQLite cannot bind an identifier, so
// the name is interpolated into DDL/DML and MUST be a plain SQL identifier.
var validTable = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*$`)

// Cache is a SQLite-backed key/value store that persists results across runs.
type Cache struct {
	db    *sql.DB
	table string
}

// New opens (or creates) a SQLite cache at path, creating its parent directory
// when missing, backed by the given table. The table name must be a valid SQL
// identifier. Any setup failure returns an error.
func New(path, table string) (*Cache, error) {
	if !validTable.MatchString(table) {
		return nil, fmt.Errorf("kvcache: invalid table name %q", table)
	}
	// 0o700 on a directory this call creates: the cache holds decoded private
	// media (voice transcripts, image descriptions). An existing directory keeps
	// its own mode — MkdirAll does not touch it.
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, fmt.Errorf("kvcache: creating %s: %w", filepath.Dir(path), err)
	}
	db, err := sql.Open("sqlite", path)
	if err != nil {
		return nil, err
	}
	// SQLite allows a single writer; one connection avoids intra-process
	// SQLITE_BUSY and serialises sequential writes.
	db.SetMaxOpenConns(1)
	// This is the first statement to touch the file, so it is where an unusable
	// path surfaces.
	if _, err := db.Exec("PRAGMA busy_timeout=5000;"); err != nil { // wait on a lock instead of erroring
		db.Close()
		return nil, fmt.Errorf("kvcache: opening %s: %w", path, err)
	}
	// journal_mode lives in the database header, so the first opener switches the
	// file to WAL (concurrent reader + writer across processes) and every later
	// connection inherits it. Between PROCESSES that switch races on an exclusive
	// lock the busy handler does not cover (measured: 7 of 8 simultaneous first
	// opens failed), so read the mode first to skip the switch on an established
	// cache, and treat a lost race as non-fatal: the winner's WAL applies to
	// everyone, and a connection that skipped the switch picks it up on its next
	// transaction. Losing the whole cache would cost far more.
	var mode string
	if err := db.QueryRow("PRAGMA journal_mode;").Scan(&mode); err == nil && !strings.EqualFold(mode, "wal") {
		_ = db.QueryRow("PRAGMA journal_mode=WAL;").Scan(&mode)
	}
	if _, err := db.Exec(fmt.Sprintf(`CREATE TABLE IF NOT EXISTS %s (
		key   TEXT PRIMARY KEY,
		value TEXT NOT NULL
	);`, table)); err != nil {
		db.Close()
		return nil, err
	}
	return &Cache{db: db, table: table}, nil
}

// Get returns the cached value for key and whether it was present. A nil cache
// reports absence.
func (c *Cache) Get(key string) (string, bool) {
	if c == nil || c.db == nil {
		return "", false
	}
	var value string
	if err := c.db.QueryRow(
		fmt.Sprintf("SELECT value FROM %s WHERE key = ?", c.table), key,
	).Scan(&value); err != nil {
		return "", false
	}
	return value, true
}

// Put stores key->value (insert or replace), committed immediately. A nil cache
// is a no-op.
func (c *Cache) Put(key, value string) error {
	if c == nil || c.db == nil {
		return nil
	}
	_, err := c.db.Exec(
		fmt.Sprintf("INSERT INTO %s(key, value) VALUES(?, ?) "+
			"ON CONFLICT(key) DO UPDATE SET value = excluded.value", c.table),
		key, value,
	)
	return err
}

// Close checkpoints the WAL into the main file and closes the database. Safe to
// call on a nil cache.
func (c *Cache) Close() error {
	if c == nil || c.db == nil {
		return nil
	}
	// Fold the WAL back so the cache is a single file at rest.
	c.db.Exec("PRAGMA wal_checkpoint(TRUNCATE);")
	return c.db.Close()
}

// Signature returns the first 10 hex chars of SHA-1 of the parts joined by "|".
func Signature(parts ...string) string {
	h := sha1.Sum([]byte(strings.Join(parts, "|")))
	return fmt.Sprintf("%x", h)[:10]
}
