package media

import (
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/rs/zerolog"
	"tg2llm/pkg/ctxlog"

	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/infra/kvcache"
)

// notIncluded is the placeholder text AyuGram uses for unavailable files.
const notIncluded = "File not included"

// defaultVisionPrompt is the Russian prompt sent to the vision model by default.
const defaultVisionPrompt = "Опиши, что изображено на картинке, и передай весь важный текст с неё. " +
	"Если это мем или шутка — кратко объясни смысл; если нет — про мемы не упоминай. " +
	"Ответь одним абзацем, без переносов строк."

// defaultFramePrompt describes the frame of a video note. It drops the photo
// prompt's "transfer all the text" and meme clauses: a video note is a talking
// head, and demanding text or a joke makes the model pad every description with
// a sentence about what the frame does not contain.
const defaultFramePrompt = "Опиши одним абзацем, что видно в кадре видеосообщения: кто в кадре и что вокруг. " +
	"Не упоминай то, чего в кадре нет. Ответь без переносов строк."

const (
	kindPhoto = "photo"
	kindVoice = "voice"
	kindFrame = "frame"

	// Telegram export "media_type" values this decoder acts on.
	mediaVoice     = "voice_message"
	mediaVideoNote = "video_message" // round video note
)

// TranscribeFunc is the signature for a voice transcription backend.
type TranscribeFunc func(absPath string) string

// DescribeFunc is the signature for a photo description backend.
type DescribeFunc func(absPath string) string

// DescribeFrameFunc is the signature for a video-note frame description backend.
// It is distinct from DescribeFunc so the two describers cannot be transposed at a
// call site: they carry different prompts and different cache salts.
type DescribeFrameFunc func(absPath string) string

// Decoder implements app.MediaDecoder. It resolves media files relative to the export root
// and decodes them using injected backends (transcribeFn for voice and video-note audio,
// describeFn for photos, describeFrameFn for video-note frames).
type Decoder struct {
	exportRoot      string
	transcribeFn    TranscribeFunc
	describeFn      DescribeFunc
	describeFrameFn DescribeFrameFunc
	cache           *kvcache.Cache
	cacheVersions   map[string]string
	logger          zerolog.Logger

	mu       sync.Mutex
	inflight map[string]*inflightCall
}

// NewDecoder creates a media decoder. Any of the backends or the cache may be nil to
// disable the corresponding functionality.
func NewDecoder(exportRoot string, transcribeFn TranscribeFunc, describeFn DescribeFunc, describeFrameFn DescribeFrameFunc, cache *kvcache.Cache, cacheVersions map[string]string, logger zerolog.Logger) *Decoder {
	if cacheVersions == nil {
		cacheVersions = make(map[string]string)
	}
	logger = logger.With().Str("component", "media-decoder").Logger()
	return &Decoder{
		exportRoot:      exportRoot,
		transcribeFn:    transcribeFn,
		describeFn:      describeFn,
		describeFrameFn: describeFrameFn,
		cache:           cache,
		cacheVersions:   cacheVersions,
		logger:          logger,
		inflight:        make(map[string]*inflightCall),
	}
}

// Decode attempts to decode a media message. Returns "" if no decoding is applicable or fails.
// The media_type cases come first: they name the medium exactly, while HasPhoto is the
// fallback for a message carrying only a "photo" key.
func (d *Decoder) Decode(msg dialogue.Message) string {
	switch {
	case msg.MediaType == mediaVoice && d.transcribeFn != nil:
		d.traceRequest(msg)
		return d.decodePath(msg.FilePath, d.transcribeFn, kindVoice)
	case msg.MediaType == mediaVideoNote && (d.transcribeFn != nil || d.describeFrameFn != nil):
		d.traceRequest(msg)
		return d.decodeVideoNote(msg)
	case msg.HasPhoto && d.describeFn != nil:
		d.traceRequest(msg)
		return d.decodePath(msg.PhotoPath, d.describeFn, kindPhoto)
	}
	return ""
}

// traceRequest is called per decoded branch rather than once at entry so a plain
// text message never pays for building a sub-logger.
func (d *Decoder) traceRequest(msg dialogue.Message) {
	log := ctxlog.Op(d.logger, "Decoder.Decode")
	log.Trace().Object("msg", msg).Msg("decode requested")
}

// decodeVideoNote decodes a round video message: whisper transcribes the audio
// track of the video file and the vision backend describes the frame Telegram
// exported as its thumbnail. Each half is keyed under its own backend's kind, so
// enabling vision later still reuses the transcripts already cached.
func (d *Decoder) decodeVideoNote(msg dialogue.Message) string {
	var transcript, desc string
	if d.transcribeFn != nil {
		transcript = d.decodePath(msg.FilePath, d.transcribeFn, kindVoice)
	}
	if d.describeFrameFn != nil {
		desc = d.decodePath(msg.ThumbnailPath, d.describeFrameFn, kindFrame)
	}
	return CombineVideo(desc, transcript)
}

func (d *Decoder) decodePath(rel string, fn func(string) string, kind string) string {
	ctxLog := ctxlog.Op(d.logger, "Decoder.decodePath").With().Str("kind", kind).Str("path", rel).Logger()

	if rel == "" || strings.Contains(rel, notIncluded) {
		return ""
	}
	absPath := filepath.Join(d.exportRoot, rel)
	hash, err := hashFile(absPath)
	if err != nil {
		ctxLog.Warn().Err(err).Msg("missing media file")
		return ""
	}

	// Key on the file's content hash so identical media (even under a different
	// path) is decoded once; the version salt re-decodes when the model changes.
	key := fmt.Sprintf("%s:%s", kind, hash)
	if version := d.cacheVersions[kind]; version != "" {
		key = fmt.Sprintf("%s:%s", key, version)
	}

	if d.cache != nil {
		if hit, ok := d.cache.Get(key); ok {
			return hit
		}
	}

	return d.decodeOnce(key, absPath, fn, ctxLog)
}

// inflightCall tracks an in-progress decode so concurrent requests for the same
// cache key share one backend call (and one identical result) instead of racing.
type inflightCall struct {
	wg  sync.WaitGroup
	val string
}

// decodeOnce runs fn for key at most once across concurrent callers (singleflight):
// the first caller decodes and caches while later callers with the same key wait
// and reuse its result, so parallel workers never decode identical media twice.
func (d *Decoder) decodeOnce(key, absPath string, fn func(string) string, ctxLog zerolog.Logger) string {
	d.mu.Lock()
	if c, ok := d.inflight[key]; ok {
		d.mu.Unlock()
		c.wg.Wait()
		return c.val
	}
	c := &inflightCall{}
	c.wg.Add(1)
	d.inflight[key] = c
	d.mu.Unlock()

	// Releasing the slot and waking waiters runs in a defer so a panic in fn can
	// never deadlock concurrent callers or leak the key.
	defer func() {
		d.mu.Lock()
		delete(d.inflight, key)
		d.mu.Unlock()
		c.wg.Done()
	}()

	ctxLog.Info().Msg("decoding media")
	text := strings.TrimSpace(fn(absPath))
	if text != "" && d.cache != nil {
		if err := d.cache.Put(key, text); err != nil {
			ctxLog.Warn().Err(err).Msg("cache put failed")
		}
	}
	c.val = text
	return text
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

// Flush finalizes the cache. With SQLite each Put is already durable, so this
// only checkpoints the WAL and closes the database at end of run.
func (d *Decoder) Flush() error {
	if d.cache != nil {
		return d.cache.Close()
	}
	return nil
}

// WhisperConfig configures the voice-transcription backend (whisper.cpp).
type WhisperConfig struct {
	Model string
	Bin   string
	Lang  string
}

// VisionConfig configures the photo-description backend (OpenAI-compatible vision server).
type VisionConfig struct {
	Model  string
	Host   string
	Prompt string
}

// DefaultCachePath is the decoded-media database inside the shared cache
// directory. Every command and every export share it: media is keyed by content
// hash, so a file already decoded for one run is never decoded again.
func DefaultCachePath() string {
	return kvcache.SharedPath("media.db")
}

// BuildDefaultDecoder wires whisper.cpp + the vision backend into a Decoder. A nil
// whisper (or one with an empty Model) disables voice transcription; a nil vision
// disables photo and video-note-frame description.
func BuildDefaultDecoder(exportRoot, cachePath string, whisper *WhisperConfig, vision *VisionConfig, logger zerolog.Logger) *Decoder {
	var transcribeFn TranscribeFunc
	var describeFn DescribeFunc
	var describeFrameFn DescribeFrameFunc
	cacheVersions := map[string]string{}

	if whisper != nil && whisper.Model != "" {
		t := NewTranscriber(whisper.Model, whisper.Bin, whisper.Lang, "ffmpeg", 600, logger)
		transcribeFn = t.Transcribe
		cacheVersions[kindVoice] = kvcache.Signature(filepath.Base(whisper.Model), whisper.Lang)
	}

	if vision != nil {
		// An explicit --vision-prompt drives both media; only the defaults differ,
		// because a video-note frame answers a different question than a photo.
		photoPrompt, framePrompt := vision.Prompt, vision.Prompt
		if vision.Prompt == "" {
			photoPrompt, framePrompt = defaultVisionPrompt, defaultFramePrompt
		}
		describeFn = NewOpenAIDescriber(vision.Model, vision.Host, photoPrompt, logger).Describe
		describeFrameFn = NewOpenAIDescriber(vision.Model, vision.Host, framePrompt, logger).Describe
		cacheVersions[kindPhoto] = kvcache.Signature(vision.Model, photoPrompt)
		cacheVersions[kindFrame] = kvcache.Signature(vision.Model, framePrompt)
	}

	// The cache is optional: on any open error decoding simply proceeds uncached.
	// (Contrast NewWriter, which returns an error because a missing output sink
	// is fatal.)
	var cache *kvcache.Cache
	if cachePath != "" {
		ctxLog := ctxlog.Op(logger, "BuildDefaultDecoder").With().Str("path", cachePath).Logger()
		c, err := kvcache.New(cachePath, "media_cache")
		if err != nil {
			ctxLog.Warn().Err(err).Msg("cache open failed")
		} else {
			cache = c
			// The path is resolved from $TG2LLM_CACHE_DIR / the checkout / the user
			// cache dir, so log which one won: a surprise full re-decode is otherwise
			// invisible.
			ctxLog.Info().Msg("decoded-media cache")
		}
	}

	return NewDecoder(exportRoot, transcribeFn, describeFn, describeFrameFn, cache, cacheVersions, logger)
}
