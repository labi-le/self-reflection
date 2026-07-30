package media

import "testing"

// TestCombineVideo pins the exact join used for every video: the export decoder
// renders video notes with it and internal/bot/botmedia renders inbound videos
// with it, so a separator or ordering change is a visible product change.
func TestCombineVideo(t *testing.T) {
	cases := []struct {
		name       string
		desc       string
		transcript string
		want       string
	}{
		{"both halves", "кадр", "речь", "кадр Речь: речь"},
		{"frame only", "кадр", "", "кадр"},
		// The marker rides along even with no description, so its absence can only
		// mean "description only" — the persona rendering relies on that.
		{"transcript only", "", "речь", "Речь: речь"},
		{"nothing decoded", "", "", ""},
		// A description that quotes the marker is neutralised, so the separator stays
		// unambiguous and the persona rendering cannot mistake machine prose for speech.
		{"description quoting the marker", "на кадре Речь: привет", "ага", "на кадре речь: привет Речь: ага"},
		{"description quoting the marker, no speech", "на кадре Речь: привет", "", "на кадре речь: привет"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := CombineVideo(c.desc, c.transcript); got != c.want {
				t.Fatalf("CombineVideo(%q, %q) = %q, want %q", c.desc, c.transcript, got, c.want)
			}
		})
	}
}
