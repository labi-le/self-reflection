package store_test

import (
	"path/filepath"
	"testing"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot/store"
)

func openStore(t *testing.T) *store.Store {
	t.Helper()
	s := store.New(filepath.Join(t.TempDir(), "cm.db"), 0, zerolog.Nop())
	if s == nil {
		t.Fatal("store.New returned nil for a valid temp path")
	}
	t.Cleanup(func() { _ = s.Close() })
	return s
}

// Test_Profile_roundtrips proves the profile upserts and reads back with its
// build-count, and is empty before anything is stored.
func Test_Profile_roundtrips(t *testing.T) {
	s := openStore(t)

	if p, c, err := s.GetProfile(7); err != nil || p != "" || c != 0 {
		t.Fatalf("empty profile = (%q, %d, %v), want (\"\", 0, nil)", p, c, err)
	}

	if err := s.SaveProfile(7, "зовут Вадик", 42); err != nil {
		t.Fatalf("SaveProfile: %v", err)
	}
	if p, c, err := s.GetProfile(7); err != nil || p != "зовут Вадик" || c != 42 {
		t.Fatalf("GetProfile = (%q, %d, %v)", p, c, err)
	}

	if err := s.SaveProfile(7, "зовут Вадик, программист", 50); err != nil {
		t.Fatalf("SaveProfile upsert: %v", err)
	}
	if p, c, _ := s.GetProfile(7); p != "зовут Вадик, программист" || c != 50 {
		t.Fatalf("after upsert = (%q, %d), want the new profile at count 50", p, c)
	}
}

// Test_CountMessages_and_LoadFirst proves the message count and that LoadFirst
// returns the oldest turns in order.
func Test_CountMessages_and_LoadFirst(t *testing.T) {
	s := openStore(t)
	msgs := []string{"привет", "я вадик", "как дела", "норм", "что нового"}
	for i, txt := range msgs {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		if err := s.AppendMessage(9, role, txt, int64(i+1)); err != nil {
			t.Fatalf("AppendMessage: %v", err)
		}
	}

	if n, err := s.CountMessages(9); err != nil || n != 5 {
		t.Fatalf("CountMessages = (%d, %v), want 5", n, err)
	}

	first, err := s.LoadFirst(9, 2)
	if err != nil {
		t.Fatalf("LoadFirst: %v", err)
	}
	if len(first) != 2 || first[0].Content != "привет" || first[1].Content != "я вадик" {
		t.Fatalf("LoadFirst(2) = %+v, want the two oldest in order", first)
	}
}
