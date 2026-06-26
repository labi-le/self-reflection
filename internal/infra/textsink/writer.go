package textsink

import (
	"fmt"
	"os"

	"tg2llm/internal/domain/dialogue"
)

// Writer implements app.DialogueSink to write formatted dialogue to a text file.
type Writer struct {
	file *os.File
}

// NewWriter creates a new file-backed Writer.
func NewWriter(path string) (*Writer, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("opening output %s: %w", path, err)
	}
	return &Writer{file: f}, nil
}

// WriteDayHeader writes a day-separator line, e.g. "\n=== YYYY-MM-DD ===\n".
// The leading \n is intentional (a blank line before each day header).
func (w *Writer) WriteDayHeader(date string) error {
	_, err := fmt.Fprintf(w.file, "\n=== %s ===\n", date)
	return err
}

// WriteMessage writes a single formatted message line.
// Format: [id] [HH:MM:SS] sender(reply_str): text\n
func (w *Writer) WriteMessage(msg dialogue.Message, formattedText string) error {
	timeStr := msg.Date.Format("15:04:05")
	replyStr := ""
	if msg.ReplyToMsgID != nil {
		replyStr = fmt.Sprintf(" (reply to %d)", *msg.ReplyToMsgID)
	}
	_, err := fmt.Fprintf(w.file, "[%d] [%s] %s%s: %s\n", msg.ID, timeStr, msg.Sender, replyStr, formattedText)
	return err
}

// Close closes the underlying file.
func (w *Writer) Close() error {
	return w.file.Close()
}

// Path returns the output file path.
func (w *Writer) Path() string {
	return w.file.Name()
}
