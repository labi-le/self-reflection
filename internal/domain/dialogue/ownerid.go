package dialogue

import "strconv"

// NormalizeOwnerID turns a bare numeric Telegram user id into the "user<N>"
// prefixed form used throughout the export ("from_id"), and passes through
// anything already prefixed (or empty). It is the single source of truth shared
// by every command that accepts a participant id on the CLI.
func NormalizeOwnerID(v string) string {
	if v == "" {
		return ""
	}
	if _, err := strconv.ParseInt(v, 10, 64); err == nil {
		return "user" + v
	}
	return v
}
