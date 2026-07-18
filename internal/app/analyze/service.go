package analyze

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/kvcache"
)

// AnalyzeService orchestrates the MAP/REDUCE portrait pipeline: deterministic
// stats, segmentation, a bounded worker pool that maps each chunk through the
// Analyzer (memoised via the content-hash cache), a pure reduce, and evidence
// index assembly. When an Embedder and Index are wired it also indexes the
// chunk texts for grounded Q&A.
type AnalyzeService struct {
	analyzer Analyzer
	embedder Embedder
	index    Index
	cache    *kvcache.Cache
	opts     AnalyzeOptions
	logger   zerolog.Logger
	now      func() time.Time
}

// NewAnalyzeService wires the ports and options for one run. cache may be nil
// (uncached); embedder/index may be nil (no Q&A index built). logger receives
// content-free warnings (e.g. a non-fatal Q&A index-build failure).
func NewAnalyzeService(a Analyzer, e Embedder, idx Index, cache *kvcache.Cache, opts AnalyzeOptions, logger zerolog.Logger) *AnalyzeService {
	return &AnalyzeService{analyzer: a, embedder: e, index: idx, cache: cache, opts: opts, logger: logger, now: time.Now}
}

type mapResult struct {
	index    int
	fragment portrait.Fragment
	analyzed bool
}

// Run executes the full pipeline and returns the assembled portrait. The result
// is byte-identical regardless of opts.Jobs: the collector writes into a
// pre-sized slice indexed by ChunkIndex, and every downstream stage is pure.
func (s *AnalyzeService) Run(ctx context.Context, msgs []dialogue.Message) (portrait.Portrait, error) {
	target := s.opts.Target
	stats := portrait.ComputeStats(msgs, target, s.opts.SessionGap)
	chunks := portrait.Segment(msgs, s.opts.SessionGap, s.opts.ChunkChars)

	frags := make([]portrait.Fragment, len(chunks))
	analyzed := make([]bool, len(chunks))
	for r := range s.mapChunks(ctx, chunks, target) {
		frags[r.index] = r.fragment
		analyzed[r.index] = r.analyzed
	}

	done := 0
	dropped := 0
	for i, ok := range analyzed {
		if ok {
			done++
		}
		dropped += frags[i].EvidenceDropped
	}
	coverage := 0.0
	if len(chunks) > 0 {
		coverage = float64(done) / float64(len(chunks))
	}

	person := portrait.Reduce(target, frags)
	dateFrom, dateTo := messageDateRange(msgs)
	meta := portrait.Meta{
		Model:             s.opts.Model,
		DateFrom:          dateFrom,
		DateTo:            dateTo,
		MessageCount:      len(msgs),
		ChunksTotal:       len(chunks),
		ChunksAnalyzed:    done,
		Coverage:          coverage,
		GeneratedAt:       s.now(),
		TokenCostEstimate: chunkCharCount(chunks),
		Disclaimer:        portrait.Disclaimer,
	}
	if s.embedder != nil && s.index != nil {
		meta.EmbedModel = s.opts.EmbedModel
	}
	var dyn *portrait.Dynamics
	if a, b, ok := s.dynamicsPair(msgs); ok {
		if da, isDyn := s.analyzer.(DynamicsAnalyzer); isDyn {
			collected, dynDropped := s.mapDynamics(ctx, da, chunks, a, b)
			dropped += dynDropped
			d := portrait.ReduceDynamics(a, b, collected)
			d.RelationshipArc = portrait.RelationshipArc(msgs, a, s.opts.SessionGap)
			dyn = &d
		}
	}
	meta.EvidenceIDsDropped = dropped
	var timeline []portrait.WindowPortrait
	if s.opts.WindowBy != "" {
		timeline = portrait.ReduceWindows(target, frags, chunks, s.opts.WindowBy)
	}
	p := portrait.Assemble(target, stats, person, dyn, timeline, msgs, meta)

	if err := s.indexChunks(ctx, chunks); err != nil {
		// Indexing is a non-fatal, opt-in Q&A convenience: never discard the
		// computed portrait when it fails. Log content-free and continue.
		s.logger.Warn().Err(err).Msg("building Q&A index; portrait written without it")
	}
	return p, nil
}

// messageDateRange returns the earliest and latest message dates in RFC3339, or
// empty strings when there are no messages. It drives meta.date_from/date_to so
// the reported window reflects the analyzed data, not the raw CLI flags.
func messageDateRange(msgs []dialogue.Message) (from, to string) {
	if len(msgs) == 0 {
		return "", ""
	}
	min, max := msgs[0].Date, msgs[0].Date
	for _, m := range msgs[1:] {
		if m.Date.Before(min) {
			min = m.Date
		}
		if m.Date.After(max) {
			max = m.Date
		}
	}
	return min.Format(time.RFC3339), max.Format(time.RFC3339)
}

// chunkCharCount is a rough, char-based token-cost estimate: the total number of
// characters across all chunk texts fed to the model (not a real token count).
func chunkCharCount(chunks []portrait.Chunk) int {
	n := 0
	for _, c := range chunks {
		n += len([]rune(c.Text))
	}
	return n
}

// mapChunks runs the feeder → workers → collector pool and streams one result
// per chunk. Results carry their ChunkIndex so the caller can place them
// order-independently.
func (s *AnalyzeService) mapChunks(ctx context.Context, chunks []portrait.Chunk, target string) <-chan mapResult {
	out := make(chan mapResult)
	if len(chunks) == 0 {
		close(out)
		return out
	}

	jobs := s.opts.Jobs
	if jobs < 1 {
		jobs = 1
	}

	feed := make(chan portrait.Chunk)
	go func() {
		for _, c := range chunks {
			feed <- c
		}
		close(feed)
	}()

	var wg sync.WaitGroup
	for range jobs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range feed {
				frag, ok := s.mapChunk(ctx, c, target)
				out <- mapResult{index: c.Index, fragment: frag, analyzed: ok}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(out)
	}()
	return out
}

// dynamicsPair reports the (a, b) participants for 1:1 dynamics. It applies only
// when Relationship is set and the pair is unambiguous: opts.With names b
// explicitly, or exactly two distinct FromIDs are present. a is always the
// target; b is the other party.
func (s *AnalyzeService) dynamicsPair(msgs []dialogue.Message) (a, b string, ok bool) {
	if !s.opts.Relationship {
		return "", "", false
	}
	a = s.opts.Target
	if s.opts.With != "" {
		return a, s.opts.With, true
	}
	seen := map[string]bool{}
	var ids []string
	for _, m := range msgs {
		if m.FromID == "" || seen[m.FromID] {
			continue
		}
		seen[m.FromID] = true
		ids = append(ids, m.FromID)
	}
	if len(ids) != 2 {
		return "", "", false
	}
	b = ids[0]
	if b == a {
		b = ids[1]
	}
	return a, b, true
}

// mapDynamics runs the same bounded, order-preserving worker pool as mapChunks
// but through the DynamicsAnalyzer. Results are collected in chunk-index order
// so the reduced dynamics are byte-identical regardless of opts.Jobs; a chunk
// whose analysis errors is skipped.
func (s *AnalyzeService) mapDynamics(ctx context.Context, da DynamicsAnalyzer, chunks []portrait.Chunk, a, b string) ([]portrait.Dynamics, int) {
	if len(chunks) == 0 {
		return nil, 0
	}
	jobs := s.opts.Jobs
	if jobs < 1 {
		jobs = 1
	}

	feed := make(chan portrait.Chunk)
	go func() {
		for _, c := range chunks {
			feed <- c
		}
		close(feed)
	}()

	type dynResult struct {
		index int
		frag  portrait.Fragment
		ok    bool
	}
	out := make(chan dynResult)
	var wg sync.WaitGroup
	for range jobs {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for c := range feed {
				frag, err := da.AnalyzeDynamics(ctx, c, a, b)
				out <- dynResult{index: c.Index, frag: frag, ok: err == nil}
			}
		}()
	}
	go func() {
		wg.Wait()
		close(out)
	}()

	frags := make([]portrait.Fragment, len(chunks))
	got := make([]bool, len(chunks))
	for r := range out {
		frags[r.index] = r.frag
		got[r.index] = r.ok
	}
	var collected []portrait.Dynamics
	dropped := 0
	for i := range chunks {
		if !got[i] {
			continue
		}
		dropped += frags[i].EvidenceDropped
		if frags[i].Dynamics != nil {
			collected = append(collected, *frags[i].Dynamics)
		}
	}
	return collected, dropped
}

// mapChunk resolves one chunk to a fragment: a cache hit skips the analyzer; a
// miss calls the analyzer and memoises a non-empty result. An analyzer error
// yields analyzed=false so the chunk is dropped from coverage.
func (s *AnalyzeService) mapChunk(ctx context.Context, c portrait.Chunk, target string) (portrait.Fragment, bool) {
	key := s.cacheKey(c)
	if v, ok := s.cache.Get(key); ok {
		var frag portrait.Fragment
		if err := json.Unmarshal([]byte(v), &frag); err == nil {
			return frag, true
		}
	}
	frag, err := s.analyzer.Analyze(ctx, c, target)
	if err != nil {
		return portrait.Fragment{}, false
	}
	if !fragmentEmpty(frag) {
		if b, err := json.Marshal(frag); err == nil {
			_ = s.cache.Put(key, string(b))
		}
	}
	return frag, true
}

func (s *AnalyzeService) cacheKey(c portrait.Chunk) string {
	sum := sha256.Sum256([]byte(c.Text))
	return kvcache.Signature("map", s.opts.Model) + ":" + hex.EncodeToString(sum[:])
}

func (s *AnalyzeService) indexChunks(ctx context.Context, chunks []portrait.Chunk) error {
	if s.embedder == nil || s.index == nil || len(chunks) == 0 {
		return nil
	}
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Text
	}
	vecs, err := s.embedder.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("embed chunks: %w", err)
	}
	if err := s.index.Add(ctx, s.collection(), chunks, vecs); err != nil {
		return fmt.Errorf("index chunks: %w", err)
	}
	return nil
}

func (s *AnalyzeService) collection() string {
	return CollectionName(s.opts.Target, s.opts.Model, s.opts.EmbedModel)
}

// CollectionName is the stable per-analysis Chroma collection id, shared by the
// indexing run and Q&A retrieval so `ask` queries exactly what `portrait` wrote.
func CollectionName(target, model, embedModel string) string {
	return "analysis-" + kvcache.Signature(target, model, embedModel)
}

// fragmentEmpty reports whether a fragment carries no inferred signal, so the
// cache is never polluted with content-free results.
func fragmentEmpty(f portrait.Fragment) bool {
	if f.Dynamics != nil {
		return false
	}
	b := f.Person.BigFive
	for _, ts := range []portrait.TraitScore{b.Openness, b.Conscientiousness, b.Extraversion, b.Agreeableness, b.Neuroticism} {
		if ts.Assessed {
			return false
		}
	}
	p := f.Person
	return len(p.Accentuations) == 0 &&
		len(p.ValuesMotivations) == 0 &&
		len(p.RecurringTopics) == 0 &&
		len(p.EmotionalPatterns) == 0 &&
		len(p.WhatToExpect) == 0 &&
		len(p.CommunicationStyle.Evidence) == 0
}
