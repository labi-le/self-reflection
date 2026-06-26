package media

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/rs/zerolog"

	"tg2llm/pkg/ctxlog"
)

// Transcriber is an adapter that transcribes audio files via ffmpeg + whisper-cli.
// It implements the func(absPath string) string signature expected by Decoder.
type Transcriber struct {
	whisperBinary string
	whisperModel  string
	language      string
	ffmpeg        string
	timeoutSecs   int
	logger        zerolog.Logger
}

// NewTranscriber creates a Transcriber with the given configuration.
func NewTranscriber(modelPath, binary, language, ffmpeg string, timeoutSecs int, logger zerolog.Logger) *Transcriber {
	if timeoutSecs <= 0 {
		timeoutSecs = 600
	}
	if ffmpeg == "" {
		ffmpeg = "ffmpeg"
	}
	if binary == "" {
		binary = "whisper-cli"
	}
	if language == "" {
		language = "ru"
	}
	return &Transcriber{
		whisperBinary: binary,
		whisperModel:  modelPath,
		language:      language,
		ffmpeg:        ffmpeg,
		timeoutSecs:   timeoutSecs,
		logger:        logger.With().Str("component", "media-transcriber").Logger(),
	}
}

// Transcribe converts absPath to 16kHz mono WAV and runs whisper-cli, returning the transcript.
// Returns "" on any error (caller treats "" as no result).
func (t *Transcriber) Transcribe(absPath string) string {
	log := ctxlog.Op(t.logger, "Transcriber.Transcribe")

	tmpDir, err := os.MkdirTemp("", "tg2llm-whisper-")
	if err != nil {
		log.Warn().Err(err).Msg("create temp dir")
		return ""
	}
	defer os.RemoveAll(tmpDir)

	wavPath := filepath.Join(tmpDir, "audio.wav")
	outBase := filepath.Join(tmpDir, "out")

	timeout := time.Duration(t.timeoutSecs) * time.Second

	// ffmpeg: convert to 16kHz mono WAV
	ffCtx, ffCancel := context.WithTimeout(context.Background(), timeout)
	defer ffCancel()
	ffCmd := exec.CommandContext(ffCtx, t.ffmpeg, "-nostdin", "-y", "-i", absPath,
		"-ar", "16000", "-ac", "1", "-f", "wav", wavPath)
	if err := ffCmd.Run(); err != nil {
		log.Warn().Err(err).Str("ffmpeg", t.ffmpeg).Str("path", absPath).Msg("ffmpeg conversion failed")
		return ""
	}

	// whisper-cli: transcribe
	whCtx, whCancel := context.WithTimeout(context.Background(), timeout)
	defer whCancel()
	whCmd := exec.CommandContext(whCtx, t.whisperBinary, "-m", t.whisperModel, "-f", wavPath,
		"-l", t.language, "-nt", "-otxt", "-of", outBase)
	if err := whCmd.Run(); err != nil {
		log.Warn().Err(err).Str("binary", t.whisperBinary).Str("model", t.whisperModel).Msg("whisper-cli failed")
		return ""
	}

	// Read output .txt
	data, err := os.ReadFile(outBase + ".txt")
	if err != nil {
		log.Warn().Err(err).Msg("read transcript file")
		return ""
	}
	return OneLine(string(data))
}
