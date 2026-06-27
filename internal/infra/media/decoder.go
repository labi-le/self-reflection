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
)

// notIncluded is the placeholder text AyuGram uses for unavailable files.
const notIncluded = "File not included"

// defaultVisionPrompt is the Russian prompt sent to the vision model by default.
const defaultVisionPrompt = "Опиши, что изображено на картинке, и передай весь важный текст с неё. " +
	"Если это мем или шутка — кратко объясни смысл; если нет — про мемы не упоминай. " +
	"Ответь одним абзацем, без переносов строк."

const (
	kindPhoto = "photo"
	kindVoice = "voice"
)

// TranscribeFunc is the signature for a voice transcription backend.
type TranscribeFunc func(absPath string) string

// DescribeFunc is the signature for a photo description backend.
type DescribeFunc func(absPath string) string

// Decoder implements app.MediaDecoder. It resolves media files relative to the export root
// and decodes them using injected backends (transcribeFn for voice, describeFn for photo).
type Decoder struct {
	exportRoot    string
	transcribeFn  TranscribeFunc
	describeFn    DescribeFunc
	cache         *Cache
	cacheVersions map[string]string
	logger        zerolog.Logger

	mu       sync.Mutex
	inflight map[string]*inflightCall
}

// NewDecoder creates a media decoder. Any of transcribeFn/describeFn/cache may be nil to disable
// the corresponding functionality.
func NewDecoder(exportRoot string, transcribeFn TranscribeFunc, describeFn DescribeFunc, cache *Cache, cacheVersions map[string]string, logger zerolog.Logger) *Decoder {
	if cacheVersions == nil {
		cacheVersions = make(map[string]string)
	}
	logger = logger.With().Str("component", "media-decoder").Logger()
	return &Decoder{
		exportRoot:    exportRoot,
		transcribeFn:  transcribeFn,
		describeFn:    describeFn,
		cache:         cache,
		cacheVersions: cacheVersions,
		logger:        logger,
		inflight:      make(map[string]*inflightCall),
	}
}

// Decode attempts to decode a media message. Returns "" if no decoding is applicable or fails.
func (d *Decoder) Decode(msg dialogue.Message) string {
	if msg.HasPhoto && d.describeFn != nil {
		log := ctxlog.Op(d.logger, "Decoder.Decode")
		log.Trace().Object("msg", msg).Msg("decode requested")
		return d.decodePath(msg.PhotoPath, d.describeFn, kindPhoto)
	}
	if msg.MediaType == "voice_message" && d.transcribeFn != nil {
		log := ctxlog.Op(d.logger, "Decoder.Decode")
		log.Trace().Object("msg", msg).Msg("decode requested")
		return d.decodePath(msg.FilePath, d.transcribeFn, kindVoice)
	}
	return ""
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
		if hit := d.cache.Get(key); hit != "" {
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
		d.cache.Put(key, text)
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

// BuildDefaultDecoder wires whisper.cpp + the vision backend into a Decoder. A nil
// whisper (or one with an empty Model) disables voice transcription; a nil vision
// disables photo description.
func BuildDefaultDecoder(exportRoot, cachePath string, whisper *WhisperConfig, vision *VisionConfig, logger zerolog.Logger) *Decoder {
	var transcribeFn TranscribeFunc
	var describeFn DescribeFunc
	cacheVersions := map[string]string{}

	if whisper != nil && whisper.Model != "" {
		t := NewTranscriber(whisper.Model, whisper.Bin, whisper.Lang, "ffmpeg", 600, logger)
		transcribeFn = t.Transcribe
		cacheVersions[kindVoice] = Signature(filepath.Base(whisper.Model), whisper.Lang)
	}

	if vision != nil {
		prompt := vision.Prompt
		if prompt == "" {
			prompt = defaultVisionPrompt
		}
		describeFn = NewOpenAIDescriber(vision.Model, vision.Host, prompt, logger).Describe
		cacheVersions[kindPhoto] = Signature(vision.Model, prompt)
	}

	// The cache is optional: NewCache returns nil on any error and decoding simply
	// proceeds uncached. (Contrast NewWriter, which returns an error because a
	// missing output sink is fatal.)
	var cache *Cache
	if cachePath != "" {
		cache = NewCache(cachePath, logger)
	}

	return NewDecoder(exportRoot, transcribeFn, describeFn, cache, cacheVersions, logger)
}
