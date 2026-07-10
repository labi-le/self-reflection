// Package store is the bot's SQLite-backed state: an allowlist of user ids the
// bot may answer and a per-chat conversation history. It reuses the media.Cache
// wrapper conventions — a single connection so writes serialise, WAL journalling,
// a busy timeout, a WAL checkpoint on Close, and nil-safe methods so a bot
// configured without a db path simply runs storeless.
package store

import (
	"database/sql"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
	"tg2llm/pkg/ctxlog"

	_ "modernc.org/sqlite"
)

// Store persists the allowlist and conversation history. The configured owner
// id is always allowed regardless of the allowed_ids table.
type Store struct {
	db            *sql.DB
	ownerID       int64
	logger        zerolog.Logger
	openWhenEmpty bool
}

// Option configures a Store at construction time.
type Option func(*Store)

// WithOpenWhenEmpty opts the store into owner-curated bootstrap: while the
// allowlist has no rows, IsAllowed permits everyone; the first AddAllowed closes
// it to owner + allowed_ids, and emptying it reopens the window. Off by default,
// so the store stays fail-closed unless this option is passed.
func WithOpenWhenEmpty() Option {
	return func(s *Store) { s.openWhenEmpty = true }
}

// schema is the create-if-not-exists DDL, run on every open so re-opening an
// existing file is idempotent.
var schema = []string{
	`CREATE TABLE IF NOT EXISTS allowed_ids (
		user_id  INTEGER PRIMARY KEY,
		note     TEXT,
		added_at INTEGER
	);`,
	`CREATE TABLE IF NOT EXISTS conversations (
		id      INTEGER PRIMARY KEY AUTOINCREMENT,
		chat_id INTEGER NOT NULL,
		role    TEXT    NOT NULL,
		content TEXT    NOT NULL,
		ts      INTEGER NOT NULL
	);`,
	`CREATE INDEX IF NOT EXISTS idx_conversations_chat ON conversations(chat_id, id);`,
	`CREATE TABLE IF NOT EXISTS chat_memory (
		chat_id        INTEGER PRIMARY KEY,
		profile        TEXT,
		built_at_count INTEGER,
		updated_at     INTEGER
	);`,
}

// New opens (or creates) the SQLite state at path, applying opts to the real
// store. Returns nil if path is empty or on any setup error, so the bot runs
// storeless (every method then no-ops and opts are not applied).
func New(path string, ownerID int64, logger zerolog.Logger, opts ...Option) *Store {
	if path == "" {
		return nil
	}
	logger = logger.With().Str("component", "bot-store").Logger()
	ctxLog := ctxlog.Op(logger, "Store.New").With().Str("path", path).Logger()
	db, err := sql.Open("sqlite", path)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("store open failed")
		return nil
	}
	// SQLite allows a single writer; one connection avoids intra-process
	// SQLITE_BUSY and serialises our writes.
	db.SetMaxOpenConns(1)
	for _, pragma := range []string{
		"PRAGMA journal_mode=WAL;",  // concurrent reader + writer across processes
		"PRAGMA busy_timeout=5000;", // wait up to 5s on a lock instead of erroring
	} {
		if _, err := db.Exec(pragma); err != nil {
			db.Close()
			ctxLog.Warn().Err(err).Str("pragma", pragma).Msg("store pragma failed")
			return nil
		}
	}
	for _, ddl := range schema {
		if _, err := db.Exec(ddl); err != nil {
			db.Close()
			ctxLog.Warn().Err(err).Msg("store schema failed")
			return nil
		}
	}
	s := &Store{db: db, ownerID: ownerID, logger: logger}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// IsAllowed reports whether the bot may answer userID: the configured owner is
// always allowed; everyone else must be present in allowed_ids. A nil store
// fails closed (returns false).
func (s *Store) IsAllowed(userID int64) bool {
	if s == nil || s.db == nil {
		return false
	}
	if s.ownerID != 0 && userID == s.ownerID {
		return true
	}
	if s.openWhenEmpty && s.allowlistEmpty() {
		return true
	}
	ctxLog := ctxlog.Op(s.logger, "Store.IsAllowed").With().Int64("user_id", userID).Logger()
	var one int
	err := s.db.QueryRow("SELECT 1 FROM allowed_ids WHERE user_id = ?", userID).Scan(&one)
	if err != nil {
		if err != sql.ErrNoRows {
			ctxLog.Warn().Err(err).Msg("allowlist lookup failed")
		}
		return false
	}
	return true
}

// allowlistEmpty reports whether allowed_ids has no rows. It fails closed: any
// query error (or a nil db) yields false, so the open-when-empty bootstrap never
// opens on a broken read.
func (s *Store) allowlistEmpty() bool {
	if s == nil || s.db == nil {
		return false
	}
	ctxLog := ctxlog.Op(s.logger, "Store.allowlistEmpty")
	var one int
	err := s.db.QueryRow("SELECT 1 FROM allowed_ids LIMIT 1").Scan(&one)
	if err != nil {
		if err != sql.ErrNoRows {
			ctxLog.Warn().Err(err).Msg("allowlist emptiness check failed")
			return false
		}
		return true
	}
	return false
}

// IsOwner reports whether userID is the configured owner. A nil store, or an
// unset owner (ownerID == 0), is never an owner.
func (s *Store) IsOwner(userID int64) bool {
	return s != nil && s.ownerID != 0 && userID == s.ownerID
}

// AddAllowed inserts (or updates the note of) userID in the allowlist. Safe to
// call on a nil store.
func (s *Store) AddAllowed(userID int64, note string) error {
	if s == nil || s.db == nil {
		return nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.AddAllowed").With().Int64("user_id", userID).Logger()
	_, err := s.db.Exec(
		"INSERT INTO allowed_ids(user_id, note, added_at) VALUES(?, ?, ?) "+
			"ON CONFLICT(user_id) DO UPDATE SET note = excluded.note",
		userID, note, time.Now().Unix(),
	)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("add allowed failed")
	}
	return err
}

// RemoveAllowed deletes userID from the allowlist (no-op if absent). Safe to
// call on a nil store.
func (s *Store) RemoveAllowed(userID int64) error {
	if s == nil || s.db == nil {
		return nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.RemoveAllowed").With().Int64("user_id", userID).Logger()
	_, err := s.db.Exec("DELETE FROM allowed_ids WHERE user_id = ?", userID)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("remove allowed failed")
	}
	return err
}

// AllowedEntry is one allowlist row: the user id and its optional note.
type AllowedEntry struct {
	UserID int64
	Note   string
}

// ListAllowed returns every allowlist entry oldest-first (by added_at). A nil
// store returns no rows.
func (s *Store) ListAllowed() ([]AllowedEntry, error) {
	if s == nil || s.db == nil {
		return nil, nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.ListAllowed")
	rows, err := s.db.Query("SELECT user_id, note FROM allowed_ids ORDER BY added_at")
	if err != nil {
		ctxLog.Warn().Err(err).Msg("list allowed failed")
		return nil, err
	}
	defer rows.Close()

	var entries []AllowedEntry
	for rows.Next() {
		var e AllowedEntry
		if err := rows.Scan(&e.UserID, &e.Note); err != nil {
			ctxLog.Warn().Err(err).Msg("scan row failed")
			return nil, err
		}
		entries = append(entries, e)
	}
	if err := rows.Err(); err != nil {
		ctxLog.Warn().Err(err).Msg("rows iteration failed")
		return nil, err
	}
	return entries, nil
}

// AppendMessage records one conversation turn for chatID. Safe to call on a nil
// store.
func (s *Store) AppendMessage(chatID int64, role, content string, ts int64) error {
	if s == nil || s.db == nil {
		return nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.AppendMessage").With().Int64("chat_id", chatID).Logger()
	_, err := s.db.Exec(
		"INSERT INTO conversations(chat_id, role, content, ts) VALUES(?, ?, ?, ?)",
		chatID, role, content, ts,
	)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("append message failed")
	}
	return err
}

// LoadRecent returns the most-recent n turns of chatID in chronological
// (oldest-first) order. A nil store, or n <= 0, returns no rows.
func (s *Store) LoadRecent(chatID int64, n int) ([]bot.Message, error) {
	if s == nil || s.db == nil || n <= 0 {
		return nil, nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.LoadRecent").With().Int64("chat_id", chatID).Logger()
	rows, err := s.db.Query(
		`SELECT role, content FROM (
			SELECT id, role, content FROM conversations
			WHERE chat_id = ?
			ORDER BY id DESC
			LIMIT ?
		) AS recent
		ORDER BY id ASC`,
		chatID, n,
	)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("load recent failed")
		return nil, err
	}
	defer rows.Close()

	var msgs []bot.Message
	for rows.Next() {
		var m bot.Message
		if err := rows.Scan(&m.Role, &m.Content); err != nil {
			ctxLog.Warn().Err(err).Msg("scan row failed")
			return nil, err
		}
		msgs = append(msgs, m)
	}
	if err := rows.Err(); err != nil {
		ctxLog.Warn().Err(err).Msg("rows iteration failed")
		return nil, err
	}
	return msgs, nil
}

// GetProfile returns the stored interlocutor profile for chatID and the message
// count it was last built from (both zero-valued if none). A nil store or a
// missing row yields ("", 0, nil).
func (s *Store) GetProfile(chatID int64) (string, int, error) {
	if s == nil || s.db == nil {
		return "", 0, nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.GetProfile").With().Int64("chat_id", chatID).Logger()
	var profile string
	var builtAt int
	err := s.db.QueryRow("SELECT profile, built_at_count FROM chat_memory WHERE chat_id = ?", chatID).Scan(&profile, &builtAt)
	if err != nil {
		if err != sql.ErrNoRows {
			ctxLog.Warn().Err(err).Msg("get profile failed")
			return "", 0, err
		}
		return "", 0, nil
	}
	return profile, builtAt, nil
}

// SaveProfile upserts the interlocutor profile for chatID, recording the message
// count it was built from (so the builder knows when to refresh). Safe on a nil store.
func (s *Store) SaveProfile(chatID int64, profile string, count int) error {
	if s == nil || s.db == nil {
		return nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.SaveProfile").With().Int64("chat_id", chatID).Logger()
	_, err := s.db.Exec(
		"INSERT INTO chat_memory(chat_id, profile, built_at_count, updated_at) VALUES(?, ?, ?, ?) "+
			"ON CONFLICT(chat_id) DO UPDATE SET profile = excluded.profile, built_at_count = excluded.built_at_count, updated_at = excluded.updated_at",
		chatID, profile, count, time.Now().Unix(),
	)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("save profile failed")
	}
	return err
}

// CountMessages returns how many conversation turns are stored for chatID (drives
// the profile-refresh cadence). A nil store returns 0.
func (s *Store) CountMessages(chatID int64) (int, error) {
	if s == nil || s.db == nil {
		return 0, nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.CountMessages").With().Int64("chat_id", chatID).Logger()
	var n int
	if err := s.db.QueryRow("SELECT COUNT(*) FROM conversations WHERE chat_id = ?", chatID).Scan(&n); err != nil {
		ctxLog.Warn().Err(err).Msg("count messages failed")
		return 0, err
	}
	return n, nil
}

// LoadFirst returns the OLDEST n turns of chatID in chronological order — the
// start of the conversation, where names and identities are usually established
// (and which the recent-window slides off as the chat grows).
func (s *Store) LoadFirst(chatID int64, n int) ([]bot.Message, error) {
	if s == nil || s.db == nil || n <= 0 {
		return nil, nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.LoadFirst").With().Int64("chat_id", chatID).Logger()
	rows, err := s.db.Query("SELECT role, content FROM conversations WHERE chat_id = ? ORDER BY id ASC LIMIT ?", chatID, n)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("load first failed")
		return nil, err
	}
	defer rows.Close()

	var msgs []bot.Message
	for rows.Next() {
		var m bot.Message
		if err := rows.Scan(&m.Role, &m.Content); err != nil {
			ctxLog.Warn().Err(err).Msg("scan row failed")
			return nil, err
		}
		msgs = append(msgs, m)
	}
	if err := rows.Err(); err != nil {
		ctxLog.Warn().Err(err).Msg("rows iteration failed")
		return nil, err
	}
	return msgs, nil
}

// ClearHistory deletes chatID's entire conversation history so the next message
// starts a fresh context. Safe to call on a nil store.
func (s *Store) ClearHistory(chatID int64) error {
	if s == nil || s.db == nil {
		return nil
	}
	ctxLog := ctxlog.Op(s.logger, "Store.ClearHistory").With().Int64("chat_id", chatID).Logger()
	_, err := s.db.Exec("DELETE FROM conversations WHERE chat_id = ?", chatID)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("clear history failed")
	}
	return err
}

// Close checkpoints the WAL into the main file and closes the database. Safe to
// call on a nil store.
func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	// Fold the WAL back so the state is a single file at rest.
	s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE);")
	return s.db.Close()
}
