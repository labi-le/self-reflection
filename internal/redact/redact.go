// Package redact scrubs secrets and personal data (PII) out of free-form text so
// it is safe to persist in a bulk training corpus or to emit from a running bot.
//
// It is a standalone, dependency-free utility (stdlib regexp only). Redact
// replaces only the matched span with Placeholder; the surrounding text is left
// byte-for-byte intact, because over-redaction corrupts training data just as
// badly as under-redaction leaks it.
//
// A note on the engine: Go's regexp is RE2, which has NO look-around and NO
// Unicode word boundary (\b is ASCII-only). Cyrillic-adjacent patterns therefore
// anchor on explicit separators / Unicode letter classes (\p{L}) instead of \b,
// and the mixed-alphanumeric token rule is expressed as a predicate over
// candidate spans rather than a look-ahead.
package redact

import "regexp"

// Placeholder is substituted for every detected secret span.
const Placeholder = "«[скрыто]»"

var (
	// email — RFC-ish local@domain.tld. ASCII throughout, so plain char classes
	// are precise enough without a boundary assertion.
	emailRe = regexp.MustCompile(`[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}`)

	// Russian phone — +7 or 8 country prefix, then a 10-digit number grouped
	// 3-3-2-2 with optional spaces, dashes or parentheses around the area code.
	// The +7/8 prefix requirement keeps it from swallowing bare digit runs.
	phoneRe = regexp.MustCompile(`(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}`)

	// Russian phone, bare contiguous 11-digit form (7XXXXXXXXXX, no '+'): phoneRe
	// requires +7/8 and the ИНН rules only match 10/12 digits, so this exact shape
	// — the observed memorised-leak format — would otherwise slip through. The \b
	// anchors keep it from firing inside a 12-digit ИНН or a longer digit run.
	phone11Re = regexp.MustCompile(`\b7\d{10}\b`)

	// password — a "пароль/password/passwd/pwd" label, a ':' or '=' separator,
	// then the value. Groups 1-3 (leading boundary, label, separator) are kept;
	// only the value (group 4) is replaced. The mandatory ':'/'=' separator is
	// what stops "пароль слишком простой" (a discussion) from being redacted,
	// and the [^\p{L}] / ^ boundary stops matching a label glued to a longer word.
	passwordRe = regexp.MustCompile(`(?i)(^|[^\p{L}])(парол[ья]|password|passwd|pwd)(\s*[:=]\s*)(\S+)`)

	// code — verification code / PIN / OTP labels. Same shape as passwordRe. The
	// ':'/'=' separator is REQUIRED (unlike a bare space) precisely because
	// "код" is an extremely common Russian word ("код города", "промокод"): only
	// "код: <value>" is treated as a secret.
	codeRe = regexp.MustCompile(`(?i)(^|[^\p{L}])(пин[\s\-]?код|pin[\s\-]?код|otp|pin|код подтверждения|код доступа|код)(\s*[:=]\s*)(\S+)`)

	// СНИЛС — ###-###-### ##. A very specific shape, so false positives are rare.
	snilsRe = regexp.MustCompile(`\b\d{3}-\d{3}-\d{3}\s\d{2}\b`)

	// Passport (RF internal) — #### ###### (4-digit series, space, 6-digit number).
	passportRe = regexp.MustCompile(`\b\d{4}\s\d{6}\b`)

	// ИНН — 12 digits (individual) or 10 digits (legal entity). The ASCII \b
	// keeps a 10-digit rule from firing inside an 11-digit phone or a longer run.
	// 12 is tried before 10 so the longer form wins. These bare-digit rules are
	// the loosest here by nature; they run last among the digit patterns.
	inn12Re = regexp.MustCompile(`\b\d{12}\b`)
	inn10Re = regexp.MustCompile(`\b\d{10}\b`)

	// JWT — three base64url segments separated by dots, starting with the
	// canonical "eyJ" header. Distinctive enough to match on shape alone.
	jwtRe = regexp.MustCompile(`\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+`)

	// AWS access key id — AKIA/ASIA followed by exactly 16 upper-case alphanums.
	// Caught explicitly because at 20 chars it is shorter than the generic
	// high-entropy threshold below.
	awsKeyRe = regexp.MustCompile(`\b(?:AKIA|ASIA)[A-Z0-9]{16}\b`)

	// High-entropy token candidate — a run of >=24 chars from the token alphabet.
	// A candidate is only redacted if it also mixes letters and digits (see
	// looksLikeSecret); the length + mixed-class guard is what keeps ordinary
	// long words, and Cyrillic text (which is outside this alphabet), untouched.
	tokenCandidateRe = regexp.MustCompile(`[A-Za-z0-9_\-]{24,}`)
)

// Redact replaces every detected secret span in s with Placeholder and returns
// the result. Ordinary text is returned unchanged. Redact is idempotent: the
// Placeholder contains characters outside every detector's alphabet, so a second
// pass over already-scrubbed text is a no-op.
//
// Order matters: label rules run first so a password/code value is scrubbed as
// such before the phone/token rules can reinterpret it; the loose bare-digit ИНН
// rules and the aggressive generic-token rule run last.
func Redact(s string) string {
	// Label rules keep the leading boundary, label and separator (groups 1-3)
	// and replace only the value.
	s = passwordRe.ReplaceAllString(s, "${1}${2}${3}"+Placeholder)
	s = codeRe.ReplaceAllString(s, "${1}${2}${3}"+Placeholder)

	// Whole-match rules, most specific first.
	s = emailRe.ReplaceAllString(s, Placeholder)
	s = jwtRe.ReplaceAllString(s, Placeholder)
	s = awsKeyRe.ReplaceAllString(s, Placeholder)
	s = phoneRe.ReplaceAllString(s, Placeholder)
	s = phone11Re.ReplaceAllString(s, Placeholder)
	s = snilsRe.ReplaceAllString(s, Placeholder)
	s = passportRe.ReplaceAllString(s, Placeholder)
	s = inn12Re.ReplaceAllString(s, Placeholder)
	s = inn10Re.ReplaceAllString(s, Placeholder)

	// Generic high-entropy tokens, guarded by a per-span predicate.
	s = tokenCandidateRe.ReplaceAllStringFunc(s, func(tok string) string {
		if looksLikeSecret(tok) {
			return Placeholder
		}
		return tok
	})
	return s
}

// hasHighEntropyToken reports whether s contains at least one span that the
// generic token rule would redact. It shares looksLikeSecret with Redact so the
// prevalence scan counts exactly what Redact would scrub.
func hasHighEntropyToken(s string) bool {
	for _, tok := range tokenCandidateRe.FindAllString(s, -1) {
		if looksLikeSecret(tok) {
			return true
		}
	}
	return false
}

// looksLikeSecret treats a long candidate as a secret only when it mixes letters
// and digits — the cheap, precise signal that separates API keys / hashes from
// ordinary long words (which never carry a digit).
func looksLikeSecret(tok string) bool {
	var hasLetter, hasDigit bool
	for _, r := range tok {
		switch {
		case r >= '0' && r <= '9':
			hasDigit = true
		case (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z'):
			hasLetter = true
		}
		if hasLetter && hasDigit {
			return true
		}
	}
	return false
}
