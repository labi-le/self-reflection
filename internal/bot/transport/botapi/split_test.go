package botapi

import "testing"

func constRand(v float64) func() float64 { return func() float64 { return v } }

func seqRand(vs ...float64) func() float64 {
	i := 0
	return func() float64 {
		v := vs[i%len(vs)]
		i++
		return v
	}
}

func Test_chunkWords_splits_on_probability(t *testing.T) {
	tests := []struct {
		name    string
		prob    float64
		maxMsgs int
		rand    func() float64
		in      string
		want    []string
	}{
		{"disabled keeps whole", 0, 0, constRand(0), "a b c", []string{"a b c"}},
		{"always break splits every word", 0.5, 0, constRand(0), "a b c", []string{"a", "b", "c"}},
		{"never break keeps whole", 0.5, 0, constRand(0.9), "a b c", []string{"a b c"}},
		{"single word stays whole", 0.9, 0, constRand(0), "hello", []string{"hello"}},
		{"cap exceeded sends whole", 1, 2, constRand(0), "a b c d", []string{"a b c d"}},
		{"mixed breaks", 0.5, 0, seqRand(0, 0.9), "one two three", []string{"one", "two three"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Given a client with scripted randomness and the split config
			c := New("tok", allowAll, testLogger, WithWordSplit(tt.prob, tt.maxMsgs), withRandFloat(tt.rand))
			// When the segment is word-chunked
			got := c.chunkWords(tt.in)
			// Then the chunks match the deterministic expectation
			if !equalStrings(got, tt.want) {
				t.Fatalf("chunkWords(%q) = %q, want %q", tt.in, got, tt.want)
			}
		})
	}
}

func Test_segments_word_splits_after_burst(t *testing.T) {
	// Given word-splitting enabled and always-breaking randomness
	c := New("tok", allowAll, testLogger, WithWordSplit(0.5, 0), withRandFloat(constRand(0)))
	// When a U+2063 burst reply whose second segment has several words is segmented
	got := c.segments("x\u2063y z")
	// Then the burst split runs first, then each segment is word-chunked
	if want := []string{"x", "y", "z"}; !equalStrings(got, want) {
		t.Fatalf("segments = %q, want %q", got, want)
	}
}
