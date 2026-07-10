package profile

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
)

type fakeStore struct {
	count      int
	profile    string
	builtAt    int
	first      []bot.Message
	recent     []bot.Message
	saved      string
	savedCount int
	savedN     int
}

func (f *fakeStore) CountMessages(int64) (int, error)             { return f.count, nil }
func (f *fakeStore) GetProfile(int64) (string, int, error)        { return f.profile, f.builtAt, nil }
func (f *fakeStore) LoadFirst(int64, int) ([]bot.Message, error)  { return f.first, nil }
func (f *fakeStore) LoadRecent(int64, int) ([]bot.Message, error) { return f.recent, nil }
func (f *fakeStore) SaveProfile(_ int64, p string, c int) error {
	f.saved, f.savedCount = p, c
	f.savedN++
	return nil
}

type fakeExtractor struct {
	out    string
	err    error
	prompt string
	calls  int
}

func (f *fakeExtractor) Extract(_ context.Context, prompt string) (string, error) {
	f.calls++
	f.prompt = prompt
	return f.out, f.err
}

func newBuilder(s Store, e Extractor, every int) *Builder {
	return NewBuilder(s, e, Config{Every: every, Logger: zerolog.Nop()})
}

// Test_Maybe_builds_profile_when_none_yet: no profile + history → extract + save,
// and the prompt carries the role-labelled transcript.
func Test_Maybe_builds_profile_when_none_yet(t *testing.T) {
	fs := &fakeStore{count: 5, recent: []bot.Message{{Role: "user", Content: "я вадик"}, {Role: "assistant", Content: "привет"}}}
	fe := &fakeExtractor{out: "— зовут Вадик"}
	b := newBuilder(fs, fe, 15)

	b.Maybe(context.Background(), 7)

	if fe.calls != 1 {
		t.Fatalf("extractor calls = %d, want 1", fe.calls)
	}
	if fs.saved != "— зовут Вадик" || fs.savedCount != 5 {
		t.Fatalf("saved = (%q, %d), want the extracted profile at count 5", fs.saved, fs.savedCount)
	}
	if !strings.Contains(fe.prompt, "собеседник: я вадик") {
		t.Fatalf("prompt missing role-labelled transcript:\n%s", fe.prompt)
	}
}

// Test_Maybe_skips_when_not_enough_new: profile exists and too few new messages → no rebuild.
func Test_Maybe_skips_when_not_enough_new(t *testing.T) {
	fs := &fakeStore{count: 20, profile: "— зовут Вадик", builtAt: 10} // 20-10=10 < every=15
	fe := &fakeExtractor{out: "x"}
	b := newBuilder(fs, fe, 15)

	b.Maybe(context.Background(), 7)

	if fe.calls != 0 || fs.savedN != 0 {
		t.Fatalf("extractor calls=%d saves=%d, want 0/0 (skip)", fe.calls, fs.savedN)
	}
}

// Test_Maybe_rebuilds_when_enough_new: enough new messages → rebuild, prompt carries the current profile.
func Test_Maybe_rebuilds_when_enough_new(t *testing.T) {
	fs := &fakeStore{count: 30, profile: "— зовут Вадик", builtAt: 10, recent: []bot.Message{{Role: "user", Content: "я программист"}}} // 30-10=20 >= 15
	fe := &fakeExtractor{out: "— зовут Вадик\n— программист"}
	b := newBuilder(fs, fe, 15)

	b.Maybe(context.Background(), 7)

	if fe.calls != 1 || fs.savedCount != 30 {
		t.Fatalf("calls=%d savedCount=%d, want 1/30", fe.calls, fs.savedCount)
	}
	if !strings.Contains(fe.prompt, "зовут Вадик") {
		t.Fatalf("prompt should feed the current profile back in:\n%s", fe.prompt)
	}
}

// Test_build_keeps_profile_on_empty_extraction: an empty extraction must NOT wipe a good profile.
func Test_build_keeps_profile_on_empty_extraction(t *testing.T) {
	fs := &fakeStore{count: 30, profile: "— зовут Вадик", builtAt: 0, recent: []bot.Message{{Role: "user", Content: "привет"}}}
	fe := &fakeExtractor{out: "   "}
	b := newBuilder(fs, fe, 15)

	b.Maybe(context.Background(), 7)

	if fs.savedN != 0 {
		t.Fatalf("saved %d times on empty extraction, want 0 (must not wipe)", fs.savedN)
	}
}

// Test_Maybe_swallows_extractor_error: an extractor failure is logged, not fatal.
func Test_Maybe_swallows_extractor_error(t *testing.T) {
	fs := &fakeStore{count: 5, recent: []bot.Message{{Role: "user", Content: "привет"}}}
	fe := &fakeExtractor{err: errors.New("boom")}
	b := newBuilder(fs, fe, 15)

	b.Maybe(context.Background(), 7) // must not panic

	if fs.savedN != 0 {
		t.Fatalf("saved on extractor error, want 0")
	}
}

// Test_buildPrompt_forbids_fabrication_and_character_judgements locks the two
// rules that stop the net-negative profile: no guessing/placeholders for unknown
// facts, and no grading of the interlocutor's character or communication style.
// A rewrite that drops either rule fails here.
func Test_buildPrompt_forbids_fabrication_and_character_judgements(t *testing.T) {
	p := strings.ToLower(buildPrompt("пока ничего", "собеседник: привет"))
	for _, rule := range []string{"не выдумывай", "не гадай", "характер"} {
		if !strings.Contains(p, rule) {
			t.Errorf("extraction prompt missing rule %q:\n%s", rule, p)
		}
	}
}
