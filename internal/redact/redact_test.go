package redact

import (
	"bufio"
	"io"
	"os"
	"testing"
)

// TestRedact drives one scenario per row: every secret category has a positive
// case (redacted to the exact expected output) and ordinary text has a negative
// case (returned byte-identical). want is the full expected output so the tests
// also lock in that only the matched span — never the whole string — is replaced.
func TestRedact(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		// ── email ──────────────────────────────────────────────────────────
		{"email is redacted", "напиши на a.b-c@mail.ru пожалуйста", "напиши на " + Placeholder + " пожалуйста"},
		{"email plus-tag is redacted", "почта user+tag@sub.example.co.uk тут", "почта " + Placeholder + " тут"},
		{"word with @ but no domain is untouched", "ник @durov в телеге", "ник @durov в телеге"},

		// ── russian phone ──────────────────────────────────────────────────
		{"phone +7 spaced is redacted", "звони +7 999 123 45 67 вечером", "звони " + Placeholder + " вечером"},
		{"phone +7 parens dashes is redacted", "тел +7 (999) 123-45-67.", "тел " + Placeholder + "."},
		{"phone 8 compact is redacted", "мой номер 89991234567 ок", "мой номер " + Placeholder + " ок"},
		{"phone 8 parens is redacted", "8 (999) 123-45-67 звонить", Placeholder + " звонить"},
		{"phone bare 7 compact is redacted", "мой номер 79001234567 ок", "мой номер " + Placeholder + " ок"},
		{"phone bare 7 standalone is redacted", "79007654321", Placeholder},

		// ── passwords / codes ──────────────────────────────────────────────
		{"parol colon value is redacted", "пароль: hunter2 не говори", "пароль: " + Placeholder + " не говори"},
		{"password equals value is redacted", "password=SuperSecret1 ok", "password=" + Placeholder + " ok"},
		{"parol with prefix word is redacted", "мой пароль = qwertyZ9 тут", "мой пароль = " + Placeholder + " тут"},
		{"otp code colon value is redacted", "код: 8421 никому", "код: " + Placeholder + " никому"},
		{"parol discussion without value is untouched", "пароль слишком простой", "пароль слишком простой"},
		{"kod goroda without separator is untouched", "код города 495 это москва", "код города 495 это москва"},
		{"promokod suffix is untouched", "промокод на скидку сегодня", "промокод на скидку сегодня"},

		// ── СНИЛС ───────────────────────────────────────────────────────────
		{"snils is redacted", "СНИЛС 112-233-445 95 готов", "СНИЛС " + Placeholder + " готов"},

		// ── passport (#### ######) ─────────────────────────────────────────
		{"passport is redacted", "паспорт 1234 567890 выдан", "паспорт " + Placeholder + " выдан"},

		// ── ИНН (10 / 12 digits) ───────────────────────────────────────────
		{"inn 12 digits is redacted", "ИНН 123456789012 физлицо", "ИНН " + Placeholder + " физлицо"},
		{"inn 10 digits is redacted", "ИНН 7707083893 юрлицо", "ИНН " + Placeholder + " юрлицо"},
		{"inn 12 starting with 7 stays fully redacted", "ИНН 712345678901 тут", "ИНН " + Placeholder + " тут"},

		// ── tokens / keys ──────────────────────────────────────────────────
		{"jwt is redacted", "токен eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.dozjgNryP4J3jVmNHl0w5N so", "токен " + Placeholder + " so"},
		{"aws access key is redacted", "ключ AKIAIOSFODNN7EXAMPLE больше", "ключ " + Placeholder + " больше"},
		{"long mixed token is redacted", "секрет aB3xK9mP2qR7sT1vW5yZ8nD4 конец", "секрет " + Placeholder + " конец"},

		// ── false-positive guards (ordinary text must be untouched) ────────
		{"age is untouched", "мне 30 лет", "мне 30 лет"},
		{"house number is untouched", "дом 5", "дом 5"},
		{"year is untouched", "в 2026 году", "в 2026 году"},
		{"money amount is untouched", "я перевёл 500 рублей", "я перевёл 500 рублей"},
		{"time of day is untouched", "встретимся в 12:30", "встретимся в 12:30"},
		{"iso date is untouched", "дата 2026-07-04 важна", "дата 2026-07-04 важна"},
		{"short digit run is untouched", "квартира 42 этаж 3", "квартира 42 этаж 3"},
		{"long cyrillic word is untouched", "оченьдлинноесловобездефисаицифр вот", "оченьдлинноесловобездефисаицифр вот"},
		{"long latin word without digit is untouched", "aaaaaaaaaaaaaaaaaaaaaaaaaaaa вот", "aaaaaaaaaaaaaaaaaaaaaaaaaaaa вот"},
		{"empty string is untouched", "", ""},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// When
			got := Redact(tt.input)

			// Then
			if got != tt.want {
				t.Fatalf("Redact(%q):\n got: %q\nwant: %q", tt.input, got, tt.want)
			}
		})
	}
}

// TestRedact_redacts_every_span_in_a_line proves multiple secrets in one string
// are each replaced independently and the ordinary text between them survives.
func TestRedact_redacts_every_span_in_a_line(t *testing.T) {
	// Given
	input := "почта a@b.ru и тел +7 999 123 45 67, пароль: topsecret9"

	// When
	got := Redact(input)

	// Then
	want := "почта " + Placeholder + " и тел " + Placeholder + ", пароль: " + Placeholder
	if got != want {
		t.Fatalf("multi-span redaction:\n got: %q\nwant: %q", got, want)
	}
}

// TestRedact_is_idempotent guards against the placeholder itself being matched
// and re-redacted on a second pass — important for runtime scrubbing that may
// run over already-scrubbed text.
func TestRedact_is_idempotent(t *testing.T) {
	// Given
	input := "email a@b.ru phone 89991234567 пароль: x9y8z7"

	// When
	once := Redact(input)
	twice := Redact(once)

	// Then
	if once != twice {
		t.Fatalf("Redact not idempotent:\n once:  %q\n twice: %q", once, twice)
	}
}

// TestPrevalenceScan is an on-demand corpus scan (skipped in the normal suite).
// Run it with:  REDACT_SCAN=1 go test -run TestPrevalenceScan -v ./internal/redact
// It reports COUNTS ONLY per category — it never logs, prints or quotes any
// matched line content, so no PII leaves the process.
func TestPrevalenceScan(t *testing.T) {
	if os.Getenv("REDACT_SCAN") == "" {
		t.Skip("set REDACT_SCAN=1 to run the finetune corpus prevalence scan")
	}

	// name → predicate. Uses the package's real detectors (white-box), so the
	// counts reflect exactly what Redact would scrub.
	categories := []struct {
		name  string
		match func(string) bool
	}{
		{"email", emailRe.MatchString},
		{"phone", phoneRe.MatchString},
		{"password", passwordRe.MatchString},
		{"code", codeRe.MatchString},
		{"snils", snilsRe.MatchString},
		{"passport", passportRe.MatchString},
		{"inn12", inn12Re.MatchString},
		{"inn10", inn10Re.MatchString},
		{"jwt", jwtRe.MatchString},
		{"aws_key", awsKeyRe.MatchString},
		{"token", hasHighEntropyToken},
	}

	for _, path := range []string{"../../finetune/data/train.jsonl", "../../finetune/data/heldout.jsonl"} {
		f, err := os.Open(path)
		if err != nil {
			t.Logf("SKIP %s: %v", path, err)
			continue
		}

		perCat := make(map[string]int, len(categories))
		var total, anyHit int
		r := bufio.NewReader(f)
		for {
			line, err := r.ReadString('\n')
			if len(line) > 0 {
				total++
				hit := false
				for _, c := range categories {
					if c.match(line) {
						perCat[c.name]++
						hit = true
					}
				}
				if hit {
					anyHit++
				}
			}
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}
		}
		_ = f.Close()

		t.Logf("=== %s: %d lines, %d with >=1 secret (%.2f%%) ===",
			path, total, anyHit, pct(anyHit, total))
		for _, c := range categories {
			t.Logf("  %-10s %6d lines (%.2f%%)", c.name, perCat[c.name], pct(perCat[c.name], total))
		}
	}
}

func pct(n, total int) float64 {
	if total == 0 {
		return 0
	}
	return 100 * float64(n) / float64(total)
}
