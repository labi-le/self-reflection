package media

import "testing"

func TestSignature(t *testing.T) {
	a := Signature("qwen2.5vl:7b", "prompt")
	if a != Signature("qwen2.5vl:7b", "prompt") {
		t.Fatal("Signature is not deterministic")
	}
	if len(a) != 10 {
		t.Fatalf("Signature length = %d, want 10", len(a))
	}
	if Signature("modelA", "p") == Signature("modelB", "p") {
		t.Error("distinct models produced identical signatures")
	}
}
