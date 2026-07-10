package botapi

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"tg2llm/internal/bot"
	"tg2llm/pkg/ctxlog"
)

const (
	getFileTimeout      = 30 * time.Second
	fileDownloadTimeout = 120 * time.Second
	mediaDecodeTimeout  = 180 * time.Second
	maxMediaBytes       = 25 << 20 // 25 MiB — the Bot API caps getFile downloads at 20 MB
)

// resolveText turns a message into the text the persona should read: the plain
// text for a text message, or the decoded transcript/description for a media
// message. Download or decode failure degrades to a caption/placeholder so a
// media message is never silently dropped.
func (c *Client) resolveText(ctx context.Context, msg *apiMessage) string {
	if msg.Sticker != nil && msg.Sticker.FileID != "" {
		return c.resolveSticker(ctx, msg.Sticker)
	}
	kind, fileID, ok := mediaRef(msg)
	if !ok || c.decoder == nil {
		return msg.Text
	}
	log := ctxlog.Op(c.logger, "Client.resolveText")
	data, err := c.downloadFile(ctx, fileID)
	if err != nil {
		log.Warn().Err(err).Msg("media download failed; using caption/placeholder")
		return formatMedia(kind, msg.Caption, "")
	}
	mctx, cancel := context.WithTimeout(ctx, mediaDecodeTimeout)
	defer cancel()
	return formatMedia(kind, msg.Caption, c.decoder.Decode(mctx, kind, data))
}

// mediaRef returns the decodable attachment's kind and file_id, or ok=false when
// the message carries no supported media.
func mediaRef(msg *apiMessage) (bot.MediaKind, string, bool) {
	switch {
	case msg.Voice != nil && msg.Voice.FileID != "":
		return bot.MediaVoice, msg.Voice.FileID, true
	case msg.Audio != nil && msg.Audio.FileID != "":
		return bot.MediaAudio, msg.Audio.FileID, true
	case len(msg.Photo) > 0:
		if id := largestPhoto(msg.Photo); id != "" {
			return bot.MediaPhoto, id, true
		}
	case msg.Video != nil && msg.Video.FileID != "":
		return bot.MediaVideo, msg.Video.FileID, true
	case msg.VideoNote != nil && msg.VideoNote.FileID != "":
		return bot.MediaVideo, msg.VideoNote.FileID, true
	case msg.Animation != nil && msg.Animation.FileID != "":
		return bot.MediaAnimation, msg.Animation.FileID, true
	case msg.Document != nil && msg.Document.FileID != "":
		if kind, ok := kindForMime(msg.Document.MimeType); ok {
			return kind, msg.Document.FileID, true
		}
	}
	return 0, "", false
}

// kindForMime maps a document's MIME type to a decoder kind; ok=false for types
// that aren't decodable media (PDFs, archives, …), which are then ignored.
func kindForMime(mime string) (bot.MediaKind, bool) {
	switch {
	case mime == "image/gif":
		return bot.MediaAnimation, true
	case strings.HasPrefix(mime, "audio/"):
		return bot.MediaAudio, true
	case strings.HasPrefix(mime, "video/"):
		return bot.MediaVideo, true
	case strings.HasPrefix(mime, "image/"):
		return bot.MediaPhoto, true
	default:
		return 0, false
	}
}

// largestPhoto picks the file_id of the highest-resolution size (max file_size).
func largestPhoto(sizes []apiPhotoSize) string {
	best := ""
	bestSize := -1
	for _, s := range sizes {
		if s.FileID != "" && s.FileSize >= bestSize {
			best, bestSize = s.FileID, s.FileSize
		}
	}
	return best
}

// downloadFile resolves fileID via getFile then downloads the bytes from the
// /file/bot… endpoint, bounded by timeouts and a size cap.
func (c *Client) downloadFile(ctx context.Context, fileID string) ([]byte, error) {
	gfCtx, cancel := context.WithTimeout(ctx, getFileTimeout)
	defer cancel()
	var gf getFileResponse
	if err := c.postJSON(gfCtx, "getFile", getFileRequest{FileID: fileID}, &gf); err != nil {
		return nil, err
	}
	if !gf.OK || gf.Result.FilePath == "" {
		return nil, fmt.Errorf("getFile: no file_path in response")
	}

	dlCtx, dlCancel := context.WithTimeout(ctx, fileDownloadTimeout)
	defer dlCancel()
	req, err := http.NewRequestWithContext(dlCtx, http.MethodGet, c.fileURL(gf.Result.FilePath), nil)
	if err != nil {
		return nil, fmt.Errorf("build file request: %s", c.sanitize(err.Error()))
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return nil, fmt.Errorf("download: %s", c.sanitize(err.Error()))
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("download: status %d", resp.StatusCode)
	}
	return io.ReadAll(io.LimitReader(resp.Body, maxMediaBytes))
}

// fileURL builds the file-download endpoint. Like methodURL it embeds the token,
// so any error carrying it must be sanitized before leaving the package.
func (c *Client) fileURL(filePath string) string {
	return c.baseURL + "/file/bot" + c.token + "/" + filePath
}

// formatMedia renders decoded media as message text. Voice becomes the raw
// transcript (it IS the user's words); photo/video become a labelled tag prefixed
// by any caption. An empty decode degrades to a bare placeholder.
func formatMedia(kind bot.MediaKind, caption, decoded string) string {
	caption = strings.TrimSpace(caption)
	decoded = strings.TrimSpace(decoded)
	switch kind {
	case bot.MediaVoice:
		if decoded != "" {
			return decoded
		}
		return "[голосовое сообщение]"
	case bot.MediaAudio:
		return withCaption(caption, mediaTag("аудио", decoded))
	case bot.MediaPhoto:
		return withCaption(caption, mediaTag("фото", decoded))
	case bot.MediaVideo:
		return withCaption(caption, mediaTag("видео", decoded))
	case bot.MediaAnimation:
		return withCaption(caption, mediaTag("гифка", decoded))
	default:
		return caption
	}
}

func mediaTag(label, decoded string) string {
	if decoded != "" {
		return "[" + label + ": " + decoded + "]"
	}
	return "[" + label + "]"
}

func withCaption(caption, tag string) string {
	if caption != "" {
		return caption + " " + tag
	}
	return tag
}

// resolveSticker turns a sticker into text: its emoji (always available) plus a
// vision description of the rendered image when one can be produced. Animated
// (TGS) stickers have no describable frame, so they degrade to the emoji alone.
func (c *Client) resolveSticker(ctx context.Context, s *apiSticker) string {
	var desc string
	if c.decoder != nil {
		data, err := c.downloadFile(ctx, s.FileID)
		if err != nil {
			log := ctxlog.Op(c.logger, "Client.resolveSticker")
			log.Warn().Err(err).Msg("sticker download failed; using emoji only")
		} else {
			mctx, cancel := context.WithTimeout(ctx, mediaDecodeTimeout)
			desc = c.decoder.Decode(mctx, bot.MediaSticker, data)
			cancel()
		}
	}
	return formatSticker(strings.TrimSpace(s.Emoji), strings.TrimSpace(desc))
}

// formatSticker combines a sticker's emoji and description into message text.
func formatSticker(emoji, desc string) string {
	switch {
	case emoji != "" && desc != "":
		return "[стикер " + emoji + ": " + desc + "]"
	case desc != "":
		return "[стикер: " + desc + "]"
	case emoji != "":
		return "[стикер " + emoji + "]"
	default:
		return "[стикер]"
	}
}
