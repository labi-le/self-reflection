package analyze_test

import (
	"context"
	"encoding/json"
	"errors"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/app/analyze"
	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/kvcache"
)

// scriptAnalyzer maps a chunk to a fragment whose openness score is derived from
// the chunk index and whose evidence is the chunk's member ids. Deterministic,
// so the reduced portrait depends only on chunk content, not scheduling order.
type scriptAnalyzer struct {
	mu     sync.Mutex
	calls  int
	failOn map[int]bool
}

func (a *scriptAnalyzer) Analyze(_ context.Context, c portrait.Chunk, target string) (portrait.Fragment, error) {
	a.mu.Lock()
	a.calls++
	a.mu.Unlock()
	if a.failOn[c.Index] {
		return portrait.Fragment{}, errors.New("boom")
	}
	ids := make([]int, 0, len(c.MemberIDs))
	for id := range c.MemberIDs {
		ids = append(ids, id)
	}
	score := 0.1 * float64(c.Index+1)
	return portrait.Fragment{
		ChunkIndex: c.Index,
		Target:     target,
		Person: portrait.PersonPortrait{
			BigFive: portrait.BigFive{
				Openness: portrait.TraitScore{Assessed: true, Score: new(score), Confidence: portrait.ConfMedium, Evidence: ids},
			},
		},
	}, nil
}

func (a *scriptAnalyzer) count() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.calls
}

type recordEmbedder struct {
	mu    sync.Mutex
	calls int
}

func (e *recordEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	e.mu.Lock()
	e.calls++
	e.mu.Unlock()
	out := make([][]float32, len(texts))
	for i := range out {
		out[i] = []float32{float32(i)}
	}
	return out, nil
}

type recordIndex struct {
	mu         sync.Mutex
	collection string
	nChunks    int
	nVecs      int
	calls      int
}

func (x *recordIndex) Add(_ context.Context, collection string, chunks []portrait.Chunk, vecs [][]float32) error {
	x.mu.Lock()
	defer x.mu.Unlock()
	x.calls++
	x.collection = collection
	x.nChunks = len(chunks)
	x.nVecs = len(vecs)
	return nil
}
func (x *recordIndex) Query(context.Context, string, []float32, int, map[string]any) ([]portrait.Chunk, error) {
	return nil, nil
}
func (x *recordIndex) Delete(context.Context, string) error { return nil }

func fixtureMsgs() []dialogue.Message {
	base := time.Date(2026, 4, 1, 9, 0, 0, 0, time.UTC)
	gap := 2 * time.Hour // > sessionGap forces separate chunks
	var msgs []dialogue.Message
	id := 1
	for s := range 3 {
		start := base.Add(time.Duration(s) * gap)
		msgs = append(msgs,
			dialogue.Message{ID: id, FromID: "user1", Sender: "A", ExtractedText: "line a", Date: start},
			dialogue.Message{ID: id + 1, FromID: "user2", Sender: "B", ExtractedText: "line b", Date: start.Add(time.Minute)},
		)
		id += 2
	}
	return msgs
}

func baseOpts() analyze.AnalyzeOptions {
	return analyze.AnalyzeOptions{
		Model:      "test-model",
		Target:     "user1",
		SessionGap: 30 * time.Minute,
		ChunkChars: 0,
		Jobs:       1,
	}
}

func TestRun_MapsEachChunkAndBuildsEvidenceIndex(t *testing.T) {
	msgs := fixtureMsgs()
	an := &scriptAnalyzer{}
	svc := analyze.NewAnalyzeService(an, nil, nil, nil, baseOpts(), zerolog.Nop())

	p, err := svc.Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if p.Meta.ChunksTotal != 3 || an.count() != 3 {
		t.Fatalf("expected 3 chunks analyzed, total=%d calls=%d", p.Meta.ChunksTotal, an.count())
	}
	if p.Meta.ChunksAnalyzed != 3 || p.Meta.Coverage != 1.0 {
		t.Fatalf("coverage wrong: analyzed=%d coverage=%v", p.Meta.ChunksAnalyzed, p.Meta.Coverage)
	}
	if p.Meta.Model != "test-model" || p.Meta.MessageCount != 6 {
		t.Fatalf("meta wrong: %+v", p.Meta)
	}
	if p.Meta.Disclaimer != portrait.Disclaimer {
		t.Fatalf("disclaimer not set")
	}
	if p.Meta.GeneratedAt.IsZero() {
		t.Fatalf("generated_at not set")
	}
	// Every id cited by the reduced portrait must resolve in evidence_index.
	cited := p.Portrait["user1"].BigFive.Openness.Evidence
	if len(cited) == 0 {
		t.Fatalf("no evidence in reduced portrait")
	}
	for _, id := range cited {
		if _, ok := p.EvidenceIndex[itoa(id)]; !ok {
			t.Fatalf("cited id %d missing from evidence_index", id)
		}
	}
}

func TestRun_DeterministicAcrossJobCounts(t *testing.T) {
	msgs := fixtureMsgs()

	run := func(jobs int) []byte {
		opts := baseOpts()
		opts.Jobs = jobs
		p, err := analyze.NewAnalyzeService(&scriptAnalyzer{}, nil, nil, nil, opts, zerolog.Nop()).Run(context.Background(), msgs)
		if err != nil {
			t.Fatalf("Run jobs=%d: %v", jobs, err)
		}
		p.Meta.GeneratedAt = time.Time{} // wall clock is the only intended nondeterminism
		b, err := json.Marshal(p)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		return b
	}

	one := run(1)
	many := run(4)
	if string(one) != string(many) {
		t.Fatalf("portrait not byte-identical across job counts\njobs=1: %s\njobs=4: %s", one, many)
	}
}

func TestRun_FailingChunkLowersCoverage(t *testing.T) {
	msgs := fixtureMsgs()
	an := &scriptAnalyzer{failOn: map[int]bool{1: true}}
	p, err := analyze.NewAnalyzeService(an, nil, nil, nil, baseOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if p.Meta.ChunksTotal != 3 || p.Meta.ChunksAnalyzed != 2 {
		t.Fatalf("expected 2/3 analyzed, got %d/%d", p.Meta.ChunksAnalyzed, p.Meta.ChunksTotal)
	}
	if p.Meta.Coverage != 2.0/3.0 {
		t.Fatalf("coverage = %v, want 2/3", p.Meta.Coverage)
	}
}

func TestRun_CacheHitSkipsAnalyzer(t *testing.T) {
	dir := t.TempDir()
	cache, err := kvcache.New(filepath.Join(dir, "cache.db"), "map")
	if err != nil {
		t.Fatalf("kvcache.New: %v", err)
	}
	defer cache.Close()

	msgs := fixtureMsgs()
	an := &scriptAnalyzer{}

	first, err := analyze.NewAnalyzeService(an, nil, nil, cache, baseOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("first Run: %v", err)
	}
	if an.count() != 3 {
		t.Fatalf("first run should call analyzer 3x, got %d", an.count())
	}

	an2 := &scriptAnalyzer{}
	second, err := analyze.NewAnalyzeService(an2, nil, nil, cache, baseOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("second Run: %v", err)
	}
	if an2.count() != 0 {
		t.Fatalf("second run should hit cache and skip analyzer, got %d calls", an2.count())
	}

	first.Meta.GeneratedAt = time.Time{}
	second.Meta.GeneratedAt = time.Time{}
	a, _ := json.Marshal(first)
	b, _ := json.Marshal(second)
	if string(a) != string(b) {
		t.Fatalf("cached run diverged from fresh run")
	}
}

func TestRun_EmbedsAndIndexesWhenEmbedderAndIndexPresent(t *testing.T) {
	msgs := fixtureMsgs()
	emb := &recordEmbedder{}
	idx := &recordIndex{}
	_, err := analyze.NewAnalyzeService(&scriptAnalyzer{}, emb, idx, nil, baseOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if idx.calls != 1 || idx.nChunks != 3 || idx.nVecs != 3 {
		t.Fatalf("index add wrong: calls=%d chunks=%d vecs=%d", idx.calls, idx.nChunks, idx.nVecs)
	}
	if idx.collection == "" {
		t.Fatalf("collection name empty")
	}
}

func itoa(i int) string { return strconv.Itoa(i) }

// dynAnalyzer is a scriptAnalyzer that also satisfies analyze.DynamicsAnalyzer,
// emitting deterministic Karpman roles and a power-balance claim per chunk with
// evidence = the chunk's member ids.
type dynAnalyzer struct {
	scriptAnalyzer
}

func (d *dynAnalyzer) AnalyzeDynamics(_ context.Context, c portrait.Chunk, a, b string) (portrait.Fragment, error) {
	ids := make([]int, 0, len(c.MemberIDs))
	for id := range c.MemberIDs {
		ids = append(ids, id)
	}
	sort.Ints(ids)
	tend := 0.1 * float64(c.Index+1)
	return portrait.Fragment{
		ChunkIndex: c.Index,
		Target:     a,
		Dynamics: &portrait.Dynamics{
			KarpmanRoles: map[string][]portrait.KarpmanRole{
				a: {{Role: "victim", Tendency: tend, Confidence: portrait.ConfMedium, Evidence: ids}},
				b: {{Role: "rescuer", Tendency: tend, Confidence: portrait.ConfMedium, Evidence: ids}},
			},
			PowerBalance: portrait.Claim{Claim: b + " leads", Confidence: portrait.ConfMedium, Evidence: ids},
		},
	}, nil
}

func relOpts() analyze.AnalyzeOptions {
	o := baseOpts()
	o.Relationship = true
	return o
}

func TestRun_PopulatesDynamicsWhenRelationshipAndPair(t *testing.T) {
	msgs := fixtureMsgs()
	svc := analyze.NewAnalyzeService(&dynAnalyzer{}, nil, nil, nil, relOpts(), zerolog.Nop())
	p, err := svc.Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if p.Dynamics == nil {
		t.Fatalf("dynamics not populated for relationship + 2-sender scope")
	}
	roles := p.Dynamics.KarpmanRoles["user1"]
	if len(roles) == 0 {
		t.Fatalf("no karpman roles for target")
	}
	if len(p.Dynamics.RelationshipArc) == 0 {
		t.Fatalf("relationship arc empty")
	}
	if p.Dynamics.PowerBalance.Claim == "" {
		t.Fatalf("power balance empty")
	}
	// Every dynamics-cited id must resolve in the evidence index.
	for _, r := range roles {
		for _, id := range r.Evidence {
			if _, ok := p.EvidenceIndex[itoa(id)]; !ok {
				t.Fatalf("dynamics id %d missing from evidence_index", id)
			}
		}
	}
}

func TestRun_NilDynamicsWhenNotRelationship(t *testing.T) {
	msgs := fixtureMsgs()
	p, err := analyze.NewAnalyzeService(&dynAnalyzer{}, nil, nil, nil, baseOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if p.Dynamics != nil {
		t.Fatalf("dynamics must be nil without Relationship")
	}
}

func TestRun_NilDynamicsWhenAnalyzerLacksCapability(t *testing.T) {
	msgs := fixtureMsgs()
	// scriptAnalyzer is not a DynamicsAnalyzer.
	p, err := analyze.NewAnalyzeService(&scriptAnalyzer{}, nil, nil, nil, relOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if p.Dynamics != nil {
		t.Fatalf("dynamics must be nil when analyzer is not a DynamicsAnalyzer")
	}
}

func TestRun_NilDynamicsWhenNotPair(t *testing.T) {
	msgs := fixtureMsgs()
	// A third distinct sender makes the pair ambiguous.
	msgs = append(msgs, dialogue.Message{
		ID: 99, FromID: "user3", Sender: "C", ExtractedText: "line c",
		Date: time.Date(2026, 4, 2, 9, 0, 0, 0, time.UTC),
	})
	p, err := analyze.NewAnalyzeService(&dynAnalyzer{}, nil, nil, nil, relOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if p.Dynamics != nil {
		t.Fatalf("dynamics must be nil for a non-pair (3-sender) scope")
	}
}

func TestRun_DynamicsDeterministicAcrossJobCounts(t *testing.T) {
	msgs := fixtureMsgs()
	run := func(jobs int) []byte {
		opts := relOpts()
		opts.Jobs = jobs
		p, err := analyze.NewAnalyzeService(&dynAnalyzer{}, nil, nil, nil, opts, zerolog.Nop()).Run(context.Background(), msgs)
		if err != nil {
			t.Fatalf("Run jobs=%d: %v", jobs, err)
		}
		p.Meta.GeneratedAt = time.Time{}
		b, err := json.Marshal(p)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		return b
	}
	if string(run(1)) != string(run(4)) {
		t.Fatalf("dynamics portrait not byte-identical across job counts")
	}
}

type failIndex struct{}

func (failIndex) Add(context.Context, string, []portrait.Chunk, [][]float32) error {
	return errors.New("boom")
}
func (failIndex) Query(context.Context, string, []float32, int, map[string]any) ([]portrait.Chunk, error) {
	return nil, nil
}
func (failIndex) Delete(context.Context, string) error { return nil }

// TestRun_IndexFailureIsNonFatal locks fix 4: a failed Q&A index build must not
// discard the computed portrait; Run returns it with a nil error.
func TestRun_IndexFailureIsNonFatal(t *testing.T) {
	msgs := fixtureMsgs()
	svc := analyze.NewAnalyzeService(&scriptAnalyzer{}, &recordEmbedder{}, failIndex{}, nil, baseOpts(), zerolog.Nop())
	p, err := svc.Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("index failure must not fail Run: %v", err)
	}
	if p.Meta.ChunksTotal != 3 || len(p.Portrait) == 0 {
		t.Fatalf("portrait should still be assembled despite index failure: %+v", p.Meta)
	}
}

type droppingAnalyzer struct{}

func (droppingAnalyzer) Analyze(_ context.Context, c portrait.Chunk, target string) (portrait.Fragment, error) {
	return portrait.Fragment{ChunkIndex: c.Index, Target: target, EvidenceDropped: 2}, nil
}

// TestRun_SumsEvidenceDropped locks fix 2: Run sums each fragment's
// EvidenceDropped into meta.evidence_ids_dropped.
func TestRun_SumsEvidenceDropped(t *testing.T) {
	msgs := fixtureMsgs()
	p, err := analyze.NewAnalyzeService(droppingAnalyzer{}, nil, nil, nil, baseOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if p.Meta.EvidenceIDsDropped != 6 { // 3 chunks * 2 dropped each
		t.Fatalf("EvidenceIDsDropped = %d, want 6", p.Meta.EvidenceIDsDropped)
	}
}

// TestRun_MetaDatesAndTokenEstimate locks fix 5: date_from/date_to derive from
// the min/max message date (RFC3339) and token_cost_estimate reflects the total
// chunk character count.
func TestRun_MetaDatesAndTokenEstimate(t *testing.T) {
	msgs := fixtureMsgs()
	p, err := analyze.NewAnalyzeService(&scriptAnalyzer{}, nil, nil, nil, baseOpts(), zerolog.Nop()).Run(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	from, err := time.Parse(time.RFC3339, p.Meta.DateFrom)
	if err != nil {
		t.Fatalf("date_from not RFC3339 (%q): %v", p.Meta.DateFrom, err)
	}
	to, err := time.Parse(time.RFC3339, p.Meta.DateTo)
	if err != nil {
		t.Fatalf("date_to not RFC3339 (%q): %v", p.Meta.DateTo, err)
	}
	if !to.After(from) {
		t.Fatalf("date_to %v not after date_from %v", to, from)
	}
	if p.Meta.TokenCostEstimate <= 0 {
		t.Fatalf("token_cost_estimate not set: %d", p.Meta.TokenCostEstimate)
	}
}
