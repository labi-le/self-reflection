package analyze

import (
	"context"
	"testing"

	"tg2llm/internal/domain/portrait"
)

// Compile-time interface-satisfaction guards mirror internal/app/ports_test.go's
// intent: the fakes below are the shape real adapters must implement.
var (
	_ Analyzer = (*fakeAnalyzer)(nil)
	_ Embedder = (*fakeEmbedder)(nil)
	_ Index    = (*fakeIndex)(nil)
)

type fakeAnalyzer struct{}

func (fakeAnalyzer) Analyze(_ context.Context, c portrait.Chunk, target string) (portrait.Fragment, error) {
	return portrait.Fragment{ChunkIndex: c.Index, Target: target}, nil
}

type fakeEmbedder struct{}

func (fakeEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	return make([][]float32, len(texts)), nil
}

type fakeIndex struct{}

func (fakeIndex) Add(context.Context, string, []portrait.Chunk, [][]float32) error {
	return nil
}
func (fakeIndex) Query(context.Context, string, []float32, int, map[string]any) ([]portrait.Chunk, error) {
	return nil, nil
}
func (fakeIndex) Delete(context.Context, string) error { return nil }

func TestFakesSatisfyPorts(t *testing.T) {
	f := fakeAnalyzer{}
	frag, err := f.Analyze(context.Background(), portrait.Chunk{Index: 2}, "user1")
	if err != nil || frag.ChunkIndex != 2 || frag.Target != "user1" {
		t.Fatalf("fake analyzer round-trip failed: %+v err=%v", frag, err)
	}
}
