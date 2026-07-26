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
		{"transcript only", "", "речь", "речь"},
		{"nothing decoded", "", "", ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := CombineVideo(c.desc, c.transcript); got != c.want {
				t.Fatalf("CombineVideo(%q, %q) = %q, want %q", c.desc, c.transcript, got, c.want)
			}
		})
	}
}
