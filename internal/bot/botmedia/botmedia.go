// Package botmedia decodes Telegram media (voice, photo, video) into text for the
// persona: a whisper.cpp transcript for voice and video audio, and an
// OpenAI-compatible vision description for a photo or a video's first frame. It
// implements bot.MediaDecoder and reuses the internal/infra/media primitives; the
// transport owns the Telegram-specific download and hands raw bytes here.
package botmedia

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/internal/bot"
	"tg2llm/internal/infra/media"
	"tg2llm/pkg/ctxlog"
)

var _ bot.MediaDecoder = (*Decoder)(nil)

const frameTimeout = 60 * time.Second

// defaultVisionPrompt is the Russian prompt used when none is configured. It asks
// for a one-paragraph description plus any text, covering both photos and frames.
const defaultVisionPrompt = "Опиши, что изображено на картинке или кадре, и передай весь важный текст с неё. " +
	"Если это мем или шутка — кратко объясни смысл. Ответь одним абзацем, без переносов строк."

// Decoder turns media bytes into text. The three work functions are seams: New
// wires the real whisper/vision/ffmpeg backends, tests inject fakes.
type Decoder struct {
	transcribe   func(absPath string) string
	describe     func(absPath string) string
	extractFrame func(ctx context.Context, videoPath string) (string, bool)
	ffmpeg       string
	logger       zerolog.Logger
}

// Config wires the backends. An empty WhisperModel disables voice/video
// transcription; an empty VisionHost disables photo/frame description.
type Config struct {
	WhisperModel string
	WhisperBin   string
	WhisperLang  string
	VisionHost   string
	VisionModel  string
	VisionPrompt string
	FFmpeg       string
	TimeoutSecs  int
	Logger       zerolog.Logger
}

// New builds a Decoder from cfg. It is always safe to construct; kinds whose
// backend is absent simply decode to "".
func New(cfg Config) *Decoder {
	ffmpeg := cfg.FFmpeg
	if ffmpeg == "" {
		ffmpeg = "ffmpeg"
	}
	d := &Decoder{
		ffmpeg: ffmpeg,
		logger: cfg.Logger.With().Str("component", "bot-media").Logger(),
	}
	if cfg.WhisperModel != "" {
		d.transcribe = media.NewTranscriber(cfg.WhisperModel, cfg.WhisperBin, cfg.WhisperLang, ffmpeg, cfg.TimeoutSecs, cfg.Logger).Transcribe
	}
	if cfg.VisionHost != "" {
		prompt := cfg.VisionPrompt
		if prompt == "" {
			prompt = defaultVisionPrompt
		}
		d.describe = media.NewOpenAIDescriber(cfg.VisionModel, cfg.VisionHost, prompt, cfg.Logger).Describe
	}
	d.extractFrame = d.ffmpegFrame
	return d
}

// Decode dispatches by kind, bounded by ctx: the (blocking) backend runs in a
// goroutine so a hung whisper/vision call can never stall the caller past ctx.
func (d *Decoder) Decode(ctx context.Context, kind bot.MediaKind, data []byte) string {
	if len(data) == 0 {
		return ""
	}
	done := make(chan string, 1)
	go func() { done <- d.decode(ctx, kind, data) }()
	select {
	case out := <-done:
		return out
	case <-ctx.Done():
		log := ctxlog.Op(d.logger, "Decoder.Decode")
		log.Warn().Err(ctx.Err()).Msg("media decode timed out")
		return ""
	}
}

func (d *Decoder) decode(ctx context.Context, kind bot.MediaKind, data []byte) string {
	switch kind {
	case bot.MediaVoice:
		return d.viaTemp(data, ".ogg", d.transcribe)
	case bot.MediaAudio:
		return d.viaTemp(data, ".mp3", d.transcribe)
	case bot.MediaPhoto:
		return d.viaTemp(data, ".jpg", d.describe)
	case bot.MediaVideo:
		return d.decodeVideo(ctx, data)
	case bot.MediaSticker, bot.MediaAnimation:
		return d.frameDescribe(ctx, data)
	default:
		return ""
	}
}

// viaTemp writes data to a temp file with ext and runs fn on it, returning ""
// when fn is absent (backend disabled) or the temp write fails.
func (d *Decoder) viaTemp(data []byte, ext string, fn func(string) string) string {
	if fn == nil {
		return ""
	}
	path, cleanup, err := writeTemp(data, ext)
	if err != nil {
		log := ctxlog.Op(d.logger, "Decoder.viaTemp")
		log.Warn().Err(err).Msg("temp file")
		return ""
	}
	defer cleanup()
	return strings.TrimSpace(fn(path))
}

// decodeVideo transcribes the audio track and describes the first frame, then
// combines whichever succeeded.
func (d *Decoder) decodeVideo(ctx context.Context, data []byte) string {
	path, cleanup, err := writeTemp(data, ".mp4")
	if err != nil {
		log := ctxlog.Op(d.logger, "Decoder.decodeVideo")
		log.Warn().Err(err).Msg("temp file")
		return ""
	}
	defer cleanup()

	var transcript, desc string
	if d.transcribe != nil {
		transcript = strings.TrimSpace(d.transcribe(path))
	}
	if d.describe != nil {
		if frame, ok := d.extractFrame(ctx, path); ok {
			defer func() { _ = os.Remove(frame) }()
			desc = strings.TrimSpace(d.describe(frame))
		}
	}
	return media.CombineVideo(desc, transcript)
}

// frameDescribe renders a frame-only medium (sticker or GIF/animation) to a JPEG
// via ffmpeg's first-frame extract and describes it. It handles static WEBP,
// video WEBM and MP4/GIF; animated TGS (Lottie) has no decodable frame and yields
// "", so a sticker caller falls back to the emoji.
func (d *Decoder) frameDescribe(ctx context.Context, data []byte) string {
	if d.describe == nil {
		return ""
	}
	src, cleanup, err := writeTemp(data, ".media")
	if err != nil {
		log := ctxlog.Op(d.logger, "Decoder.frameDescribe")
		log.Warn().Err(err).Msg("temp file")
		return ""
	}
	defer cleanup()
	frame, ok := d.extractFrame(ctx, src)
	if !ok {
		return ""
	}
	defer func() { _ = os.Remove(frame) }()
	return strings.TrimSpace(d.describe(frame))
}

// ffmpegFrame extracts the first video frame to a temp JPEG, returning its path
// and true, or ("", false) on any failure.
func (d *Decoder) ffmpegFrame(ctx context.Context, videoPath string) (string, bool) {
	f, err := os.CreateTemp("", "tg2llm-frame-*.jpg")
	if err != nil {
		return "", false
	}
	frame := f.Name()
	_ = f.Close()

	fctx, cancel := context.WithTimeout(ctx, frameTimeout)
	defer cancel()
	cmd := exec.CommandContext(fctx, d.ffmpeg, "-nostdin", "-y", "-i", videoPath, "-frames:v", "1", "-q:v", "3", frame)
	if err := cmd.Run(); err != nil {
		log := ctxlog.Op(d.logger, "Decoder.ffmpegFrame")
		log.Warn().Err(err).Msg("frame extract failed")
		_ = os.Remove(frame)
		return "", false
	}
	return frame, true
}

// writeTemp writes data to a uniquely-named temp file with ext and returns its
// path plus a cleanup func that removes it.
func writeTemp(data []byte, ext string) (string, func(), error) {
	f, err := os.CreateTemp("", "tg2llm-media-*"+ext)
	if err != nil {
		return "", nil, err
	}
	name := f.Name()
	if _, err := f.Write(data); err != nil {
		_ = f.Close()
		_ = os.Remove(name)
		return "", nil, err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(name)
		return "", nil, err
	}
	return name, func() { _ = os.Remove(name) }, nil
}
