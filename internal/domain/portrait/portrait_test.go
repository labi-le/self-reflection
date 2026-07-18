package portrait

import "testing"

func TestCatalogs(t *testing.T) {
	if len(OCEAN) != 5 {
		t.Fatalf("OCEAN = %d, want 5", len(OCEAN))
	}
	if len(AccentuationTypes) != 10 {
		t.Fatalf("accentuations = %d, want 10", len(AccentuationTypes))
	}
	if len(KarpmanRoles) != 3 {
		t.Fatalf("karpman = %d, want 3", len(KarpmanRoles))
	}
}

func TestNotAssessedTraitHasNilScore(t *testing.T) {
	ts := NotAssessed()
	if ts.Assessed || ts.Score != nil {
		t.Fatalf("NotAssessed() = %+v, want assessed=false score=nil", ts)
	}
}
