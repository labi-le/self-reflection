package chatmlsink

import "testing"

func assertTurnsEqual(t *testing.T, got, want []turn) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got)=%d want %d:\n got=%+v\nwant=%+v", len(got), len(want), got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("turn %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// Two consecutive same-role turns collapse into one turn whose content is joined by
// Options.BurstDelim (no segment dropped, never concatenated without the delimiter).
func TestRepairAlternationCollapsesConsecutiveSameRole(t *testing.T) {
	s := &Sink{opts: Options{BurstDelim: burstDelim}}
	body := []turn{
		{Role: roleUser, Content: "a"},
		{Role: roleUser, Content: "b"},
		{Role: roleAssistant, Content: "c"},
	}
	got := s.repairAlternation(body)
	want := []turn{
		{Role: roleUser, Content: "a" + burstDelim + "b"},
		{Role: roleAssistant, Content: "c"},
	}
	assertTurnsEqual(t, got, want)
}

// A leading assistant turn (nothing to answer) is dropped so the body starts with
// user, then strictly alternates.
func TestRepairAlternationDropsLeadingAssistant(t *testing.T) {
	s := &Sink{opts: Options{BurstDelim: burstDelim}}
	body := []turn{
		{Role: roleAssistant, Content: "x"},
		{Role: roleUser, Content: "y"},
		{Role: roleAssistant, Content: "z"},
	}
	got := s.repairAlternation(body)
	want := []turn{
		{Role: roleUser, Content: "y"},
		{Role: roleAssistant, Content: "z"},
	}
	assertTurnsEqual(t, got, want)
}
