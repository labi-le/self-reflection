package media

import (
	"crypto/sha1"
	"fmt"
	"strings"
)

// Signature returns the first 10 hex chars of SHA-1 of the parts joined by "|".
func Signature(parts ...string) string {
	h := sha1.Sum([]byte(strings.Join(parts, "|")))
	return fmt.Sprintf("%x", h)[:10]
}
