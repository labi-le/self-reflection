package botapi

// This file holds the Telegram Bot API wire DTOs the transport marshals and
// decodes. They are deliberately minimal — only the fields the persona bot needs
// to gate and answer a message (getUpdates in, sendMessage/sendChatAction out).

// getUpdatesRequest is the long-poll request body: advance past acknowledged
// updates with Offset and hold the connection open for Timeout seconds.
type getUpdatesRequest struct {
	Offset  int64 `json:"offset"`
	Timeout int   `json:"timeout"`
}

// getUpdatesResponse is the getUpdates envelope: OK is the API-level success
// flag, Result the batch of updates.
type getUpdatesResponse struct {
	OK     bool        `json:"ok"`
	Result []apiUpdate `json:"result"`
}

// apiUpdate is one entry of a getUpdates result. Message is a pointer because
// many update kinds (edits, callbacks, joins) carry no plain message — those
// decode to nil and are skipped.
type apiUpdate struct {
	UpdateID int64       `json:"update_id"`
	Message  *apiMessage `json:"message"`
}

// apiMessage is the subset of a Telegram message the bot reads: which chat it
// belongs to, who sent it, the text, and any decodable media (voice, photo,
// video, video note) plus its caption.
type apiMessage struct {
	Chat      apiChat        `json:"chat"`
	From      *apiUser       `json:"from"`
	Text      string         `json:"text"`
	Caption   string         `json:"caption"`
	Voice     *apiFileRef    `json:"voice"`
	Audio     *apiFileRef    `json:"audio"`
	Video     *apiFileRef    `json:"video"`
	VideoNote *apiFileRef    `json:"video_note"`
	Animation *apiFileRef    `json:"animation"`
	Document  *apiDocument   `json:"document"`
	Sticker   *apiSticker    `json:"sticker"`
	Photo     []apiPhotoSize `json:"photo"`
}

type apiChat struct {
	ID int64 `json:"id"`
}

type apiUser struct {
	ID int64 `json:"id"`
}

// apiFileRef is the shared shape of single-file attachments (voice, video,
// video_note): only the file_id is needed to fetch the bytes via getFile.
type apiFileRef struct {
	FileID string `json:"file_id"`
}

// apiPhotoSize is one entry of a photo's size array; Telegram orders them
// ascending, so the largest (last, or max file_size) is the one to describe.
type apiPhotoSize struct {
	FileID   string `json:"file_id"`
	FileSize int    `json:"file_size"`
}

// apiDocument is a generic file attachment; MimeType routes it (audio/*, video/*,
// image/*) to the matching decoder — other types are ignored.
type apiDocument struct {
	FileID   string `json:"file_id"`
	MimeType string `json:"mime_type"`
}

// apiSticker is a sticker: Emoji is the emoji it represents (used even when the
// image can't be rendered); FileID is the sticker image (WEBP static, WEBM video,
// or TGS animated).
type apiSticker struct {
	FileID string `json:"file_id"`
	Emoji  string `json:"emoji"`
}

// getFileRequest / getFileResponse cover the getFile call that resolves a
// file_id to a relative file_path, which is then downloaded from the /file/bot…
// endpoint.
type getFileRequest struct {
	FileID string `json:"file_id"`
}

type getFileResponse struct {
	OK     bool        `json:"ok"`
	Result apiFileInfo `json:"result"`
}

type apiFileInfo struct {
	FilePath string `json:"file_path"`
}

// sendMessageRequest is the sendMessage body: deliver Text to ChatID.
type sendMessageRequest struct {
	ChatID int64  `json:"chat_id"`
	Text   string `json:"text"`
}

// chatActionRequest is the sendChatAction body used to show a "typing…"
// indicator before a burst segment.
type chatActionRequest struct {
	ChatID int64  `json:"chat_id"`
	Action string `json:"action"`
}
