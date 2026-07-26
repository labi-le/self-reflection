package media

import (
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"tg2llm/internal/domain/dialogue"
)

func TestDecoderPhotoAndVoice(t *testing.T) {
	root := t.TempDir()
	photoRel := writeFile(t, root, "photos/p.jpg")
	voiceRel := writeFile(t, root, "voice_messages/a.ogg")

	photoCalls, voiceCalls := 0, 0
	describe := func(string) string { photoCalls++; return "a cat on a sofa" }
	transcribe := func(string) string { voiceCalls++; return "привет мир" }

	cache := newCache(t, filepath.Join(root, "cache.db"))
	d := NewDecoder(root, transcribe, describe, nil, cache, map[string]string{"photo": "v1", "voice": "v1"}, testLogger)

	photoMsg := dialogue.Message{HasPhoto: true, PhotoPath: photoRel}
	voiceMsg := dialogue.Message{MediaType: "voice_message", FilePath: voiceRel}

	if got := d.Decode(photoMsg); got != "a cat on a sofa" {
		t.Fatalf("photo decode = %q", got)
	}
	if got := d.Decode(voiceMsg); got != "привет мир" {
		t.Fatalf("voice decode = %q", got)
	}
	if photoCalls != 1 || voiceCalls != 1 {
		t.Fatalf("calls photo=%d voice=%d, want 1/1", photoCalls, voiceCalls)
	}

	// Cache hit: second decode must not call the backend again.
	if got := d.Decode(photoMsg); got != "a cat on a sofa" {
		t.Fatalf("photo decode (cached) = %q", got)
	}
	if photoCalls != 1 {
		t.Fatalf("photo backend called %d times, want 1 (cache hit expected)", photoCalls)
	}
}

func TestDecoderMissingFile(t *testing.T) {
	root := t.TempDir()
	calls := 0
	describe := func(string) string { calls++; return "x" }
	d := NewDecoder(root, nil, describe, nil, nil, nil, testLogger)

	msg := dialogue.Message{HasPhoto: true, PhotoPath: "photos/does-not-exist.jpg"}
	if got := d.Decode(msg); got != "" {
		t.Fatalf("missing file decode = %q, want empty", got)
	}
	if calls != 0 {
		t.Fatalf("backend called %d times for missing file, want 0", calls)
	}
}

func TestDecoderNotIncludedPlaceholder(t *testing.T) {
	root := t.TempDir()
	calls := 0
	describe := func(string) string { calls++; return "x" }
	d := NewDecoder(root, nil, describe, nil, nil, nil, testLogger)

	msg := dialogue.Message{HasPhoto: true, PhotoPath: "(File not included. Change export settings.)"}
	if got := d.Decode(msg); got != "" {
		t.Fatalf("NOT_INCLUDED decode = %q, want empty", got)
	}
	if calls != 0 {
		t.Fatalf("backend called %d times for NOT_INCLUDED, want 0", calls)
	}
}

func TestDecoderEmptyResult(t *testing.T) {
	root := t.TempDir()
	rel := writeFile(t, root, "photos/p.jpg")
	describe := func(string) string { return "   " } // whitespace -> trimmed to empty
	d := NewDecoder(root, nil, describe, nil, newCache(t, filepath.Join(root, "c.db")), nil, testLogger)

	msg := dialogue.Message{HasPhoto: true, PhotoPath: rel}
	if got := d.Decode(msg); got != "" {
		t.Fatalf("empty-result decode = %q, want empty", got)
	}
}

func TestDecoderVersionSalting(t *testing.T) {
	root := t.TempDir()
	rel := writeFile(t, root, "photos/p.jpg")
	cachePath := filepath.Join(root, "cache.db")

	calls := 0
	describe := func(string) string { calls++; return "desc" }
	msg := dialogue.Message{HasPhoto: true, PhotoPath: rel}

	// Decoder A with photo version "modelA".
	dA := NewDecoder(root, nil, describe, nil, newCache(t, cachePath), map[string]string{"photo": "modelA"}, testLogger)
	dA.Decode(msg)
	if calls != 1 {
		t.Fatalf("after A: calls=%d, want 1", calls)
	}
	dA.Flush()

	// Decoder B shares the cache file but uses a different photo version -> different key -> re-decode.
	dB := NewDecoder(root, nil, describe, nil, newCache(t, cachePath), map[string]string{"photo": "modelB"}, testLogger)
	dB.Decode(msg)
	if calls != 2 {
		t.Fatalf("after B (different version): calls=%d, want 2 (no stale hit)", calls)
	}
	dB.Flush()

	// Decoder C reuses modelA version -> should hit the cache A wrote.
	dC := NewDecoder(root, nil, describe, nil, newCache(t, cachePath), map[string]string{"photo": "modelA"}, testLogger)
	dC.Decode(msg)
	if calls != 2 {
		t.Fatalf("after C (version modelA): calls=%d, want 2 (cache hit expected)", calls)
	}
	dC.Flush()
}

func TestDecoderDedupByContentHash(t *testing.T) {
	root := t.TempDir()
	relA := writeFile(t, root, "photos/a.jpg")
	relB := writeFile(t, root, "photos/b.jpg")

	calls := 0
	describe := func(string) string { calls++; return "same image" }
	d := NewDecoder(root, nil, describe, nil, newCache(t, filepath.Join(root, "c.db")), map[string]string{"photo": "v1"}, testLogger)

	if got := d.Decode(dialogue.Message{HasPhoto: true, PhotoPath: relA}); got != "same image" {
		t.Fatalf("decode A = %q", got)
	}
	if got := d.Decode(dialogue.Message{HasPhoto: true, PhotoPath: relB}); got != "same image" {
		t.Fatalf("decode B = %q", got)
	}
	if calls != 1 {
		t.Fatalf("identical content decoded %d times, want 1 (content-hash dedup)", calls)
	}
}

// TestDecoderConcurrentSingleflight verifies that many concurrent Decode calls for
// the same content hash collapse onto a single backend invocation (in-flight dedup)
// and every caller observes the same result. Run with -race.
func TestDecoderConcurrentSingleflight(t *testing.T) {
	root := t.TempDir()
	rel := writeFile(t, root, "photos/p.jpg")

	const n = 24
	var calls int32
	arrived := make(chan struct{}, 1)
	release := make(chan struct{})
	describe := func(string) string {
		atomic.AddInt32(&calls, 1)
		arrived <- struct{}{} // signal that the single in-flight backend call is parked
		<-release             // hold the in-flight slot so concurrent callers must collapse onto it
		return "one description"
	}
	d := NewDecoder(root, nil, describe, nil, newCache(t, filepath.Join(root, "c.db")), map[string]string{"photo": "v1"}, testLogger)

	msg := dialogue.Message{HasPhoto: true, PhotoPath: rel}
	results := make([]string, n)
	started := make(chan struct{}, n)
	start := make(chan struct{})
	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			started <- struct{}{}
			<-start
			results[idx] = d.Decode(msg)
		}(i)
	}
	for i := 0; i < n; i++ {
		<-started // every goroutine is scheduled and ready to race
	}
	close(start)
	<-arrived // the elected leader holds the in-flight slot; followers collapse onto it
	close(release)
	wg.Wait()

	if calls != 1 {
		t.Fatalf("backend called %d times under concurrency, want 1 (singleflight)", calls)
	}
	for i, got := range results {
		if got != "one description" {
			t.Fatalf("result[%d] = %q, want %q", i, got, "one description")
		}
	}
	if err := d.Flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}
}

func TestBuildDefaultDecoderWiring(t *testing.T) {
	d := BuildDefaultDecoder(
		t.TempDir(),
		"",
		&WhisperConfig{Model: "model.gguf", Bin: "whisper-cli", Lang: "ru"},
		&VisionConfig{Model: "qwen", Host: "http://localhost:8080"},
		testLogger,
	)
	if d.transcribeFn == nil {
		t.Error("transcribeFn should be set when whisper.Model is non-empty")
	}
	if d.describeFn == nil {
		t.Error("describeFn should be set when vision config is provided")
	}
	if d.describeFrameFn == nil {
		t.Error("describeFrameFn should be set when vision config is provided")
	}
	if d.cacheVersions["voice"] == "" {
		t.Error("voice cache version should be salted")
	}
	if d.cacheVersions["photo"] == "" {
		t.Error("photo cache version should be salted")
	}
	// The frame prompt differs from the photo prompt, so tuning one must never
	// invalidate the other's cache entries.
	if v := d.cacheVersions["frame"]; v == "" || v == d.cacheVersions["photo"] {
		t.Errorf("frame cache version = %q, want a non-empty salt distinct from photo", v)
	}

	// An explicit --vision-prompt drives both describers, so the salts collapse:
	// the frame is described exactly as the user asked, not with the built-in.
	explicit := BuildDefaultDecoder(t.TempDir(), "", nil,
		&VisionConfig{Model: "qwen", Host: "http://h", Prompt: "p"}, testLogger)
	if f, p := explicit.cacheVersions["frame"], explicit.cacheVersions["photo"]; f == "" || f != p {
		t.Errorf("explicit prompt: frame salt = %q, photo salt = %q, want one shared non-empty salt", f, p)
	}
}

func TestBuildDefaultDecoderNilConfigsDisableBackends(t *testing.T) {
	d := BuildDefaultDecoder(t.TempDir(), "", nil, nil, testLogger)
	if d.transcribeFn != nil {
		t.Error("transcribeFn should be nil when whisper config is nil")
	}
	if d.describeFn != nil || d.describeFrameFn != nil {
		t.Error("describers should be nil when vision config is nil")
	}
}

func TestBuildDefaultDecoderEmptyWhisperModelDisablesVoice(t *testing.T) {
	d := BuildDefaultDecoder(
		t.TempDir(),
		"",
		&WhisperConfig{Bin: "whisper-cli", Lang: "ru"},
		&VisionConfig{Model: "qwen", Host: "http://h", Prompt: "p"},
		testLogger,
	)
	if d.transcribeFn != nil {
		t.Error("transcribeFn should be nil when whisper.Model is empty")
	}
	if d.describeFn == nil {
		t.Error("describeFn should remain set")
	}
}

// TestDecoderVideoNote proves a round video message is decoded from both halves —
// whisper on the video's audio track, vision on the thumbnail frame Telegram
// exported beside it — and that each half is keyed under its own backend's
// version salt, so re-tuning one backend never re-runs the other.
func TestDecoderVideoNote(t *testing.T) {
	root := t.TempDir()
	videoRel := writeFileContent(t, root, "round_video_messages/v.mp4", "video-bytes")
	thumbRel := writeFileContent(t, root, "round_video_messages/v_thumb.jpg", "thumb-bytes")
	cachePath := filepath.Join(root, "cache.db")

	describeCalls, transcribeCalls := 0, 0
	describe := func(string) string { describeCalls++; return "человек на фоне окна" }
	transcribe := func(string) string { transcribeCalls++; return "привет" }
	msg := dialogue.Message{MediaType: "video_message", FilePath: videoRel, ThumbnailPath: thumbRel}
	const want = "человек на фоне окна Речь: привет"

	decodeWith := func(versions map[string]string) string {
		d := NewDecoder(root, transcribe, nil, describe, newCache(t, cachePath), versions, testLogger)
		got := d.Decode(msg)
		if err := d.Flush(); err != nil {
			t.Fatalf("flush: %v", err)
		}
		return got
	}

	if got := decodeWith(map[string]string{kindFrame: "vision1", kindVoice: "whisper1"}); got != want {
		t.Fatalf("video note decode = %q, want %q", got, want)
	}
	if describeCalls != 1 || transcribeCalls != 1 {
		t.Fatalf("calls describe=%d transcribe=%d, want 1/1", describeCalls, transcribeCalls)
	}

	if got := decodeWith(map[string]string{kindFrame: "vision1", kindVoice: "whisper1"}); got != want {
		t.Fatalf("video note decode (cached) = %q, want %q", got, want)
	}
	if describeCalls != 1 || transcribeCalls != 1 {
		t.Fatalf("after cache hit: describe=%d transcribe=%d, want 1/1 (both halves cached)", describeCalls, transcribeCalls)
	}

	// A new whisper model must re-transcribe the audio and reuse the frame.
	decodeWith(map[string]string{kindFrame: "vision1", kindVoice: "whisper2"})
	if transcribeCalls != 2 || describeCalls != 1 {
		t.Fatalf("after whisper bump: transcribe=%d describe=%d, want 2/1 (audio salted by the voice version)", transcribeCalls, describeCalls)
	}

	// A new vision prompt must re-describe the frame and reuse the transcript.
	decodeWith(map[string]string{kindFrame: "vision2", kindVoice: "whisper2"})
	if describeCalls != 2 || transcribeCalls != 2 {
		t.Fatalf("after vision bump: describe=%d transcribe=%d, want 2/2 (frame salted by the frame version)", describeCalls, transcribeCalls)
	}
}

// TestDecoderVideoNoteHalves proves each half is optional: a thumbnail the export
// left out, a file that is not on disk, or a disabled backend narrows the result
// instead of losing it, and a video note that decodes to nothing stays "" so the
// service drops it rather than emitting an empty tag.
func TestDecoderVideoNoteHalves(t *testing.T) {
	root := t.TempDir()
	videoRel := writeFileContent(t, root, "round/v.mp4", "video-bytes")
	thumbRel := writeFileContent(t, root, "round/v.jpg", "thumb-bytes")
	describe := func(string) string { return "кадр" }
	transcribe := func(string) string { return "речь" }
	silent := func(string) string { return "" }

	cases := []struct {
		name        string
		transcribe  TranscribeFunc
		describeFrm DescribeFrameFunc
		video       string
		thumb       string
		want        string
	}{
		{"both backends", transcribe, describe, videoRel, thumbRel, "кадр Речь: речь"},
		{"thumbnail absent", transcribe, describe, videoRel, "", "речь"},
		{"thumbnail not exported", transcribe, describe, videoRel, "(File not included. Change data exporting settings to download.)", "речь"},
		{"thumbnail missing on disk", transcribe, describe, videoRel, "round/gone.jpg", "речь"},
		{"video missing on disk", transcribe, describe, "round/gone.mp4", thumbRel, "кадр"},
		{"vision disabled", transcribe, nil, videoRel, thumbRel, "речь"},
		{"whisper disabled", nil, describe, videoRel, thumbRel, "кадр"},
		{"both backends silent", silent, silent, videoRel, thumbRel, ""},
		{"no backend", nil, nil, videoRel, thumbRel, ""},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d := NewDecoder(root, c.transcribe, nil, c.describeFrm, nil, nil, testLogger)
			msg := dialogue.Message{MediaType: "video_message", FilePath: c.video, ThumbnailPath: c.thumb}
			if got := d.Decode(msg); got != c.want {
				t.Fatalf("decode = %q, want %q", got, c.want)
			}
		})
	}
}

// TestDecoderVideoNoteWinsOverPhotoKey proves the media_type branch is tried
// before the HasPhoto fallback: a video note that also carries a "photo" key is
// still transcribed instead of being described from an empty photo path.
func TestDecoderVideoNoteWinsOverPhotoKey(t *testing.T) {
	root := t.TempDir()
	videoRel := writeFileContent(t, root, "round/v.mp4", "video-bytes")
	describe := func(string) string { return "не должно вызываться" }
	transcribe := func(string) string { return "речь" }

	d := NewDecoder(root, transcribe, describe, nil, nil, nil, testLogger)
	msg := dialogue.Message{MediaType: "video_message", FilePath: videoRel, HasPhoto: true}
	if got := d.Decode(msg); got != "речь" {
		t.Fatalf("decode = %q, want the transcript (video-note branch must win)", got)
	}
}
