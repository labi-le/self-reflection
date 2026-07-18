package portrait

import (
	"math"
	"testing"
	"time"

	"tg2llm/internal/domain/dialogue"
)

func approxEq(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestComputeStats_TwoSenders(t *testing.T) {
	base := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 1, FromID: "userA", Sender: "A", Date: base, ExtractedText: "привет"},
		{ID: 2, FromID: "userB", Sender: "B", Date: base.Add(2 * time.Minute), ExtractedText: "как дела?"},
		{ID: 3, FromID: "userA", Sender: "A", Date: base.Add(3 * time.Minute), ExtractedText: "норм"},
		{ID: 4, FromID: "userA", Sender: "A", Date: base.Add(90 * time.Minute), ExtractedText: "ещё тут?"},
	}
	s := ComputeStats(msgs, "userA", 30*time.Minute)
	if s.PerSender["userA"].Messages != 3 {
		t.Fatalf("A messages = %d, want 3", s.PerSender["userA"].Messages)
	}
	if s.PerSender["userB"].QuestionsRatio != 1.0 {
		t.Fatalf("B questions_ratio = %v, want 1.0", s.PerSender["userB"].QuestionsRatio)
	}
	if s.Pairwise == nil || s.Pairwise.InitiativeBalance <= 0 {
		t.Fatalf("initiative_balance should favor target A, got %+v", s.Pairwise)
	}
}

func TestComputeStats_ExactValues(t *testing.T) {
	base := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 1, FromID: "userA", Sender: "A", Date: base, ExtractedText: "привет"},
		{ID: 2, FromID: "userB", Sender: "B", Date: base.Add(2 * time.Minute), ExtractedText: "как дела?"},
		{ID: 3, FromID: "userA", Sender: "A", Date: base.Add(3 * time.Minute), ExtractedText: "норм"},
		{ID: 4, FromID: "userA", Sender: "A", Date: base.Add(90 * time.Minute), ExtractedText: "ещё тут?"},
	}
	s := ComputeStats(msgs, "userA", 30*time.Minute)

	a := s.PerSender["userA"]
	if a.DisplayName != "A" {
		t.Errorf("A display_name = %q, want A", a.DisplayName)
	}
	if a.Chars != 18 {
		t.Errorf("A chars = %d, want 18", a.Chars)
	}
	if !approxEq(a.AvgMsgLen, 6.0) {
		t.Errorf("A avg_msg_len = %v, want 6.0", a.AvgMsgLen)
	}
	if a.MedianMsgLen != 6 {
		t.Errorf("A median_msg_len = %d, want 6", a.MedianMsgLen)
	}
	if !approxEq(a.QuestionsRatio, 1.0/3.0) {
		t.Errorf("A questions_ratio = %v, want 1/3", a.QuestionsRatio)
	}
	if a.Initiations != 2 {
		t.Errorf("A initiations = %d, want 2", a.Initiations)
	}
	if a.ConversationEnders != 2 {
		t.Errorf("A conversation_enders = %d, want 2", a.ConversationEnders)
	}
	if a.ActiveHoursHistogram[9] != 2 || a.ActiveHoursHistogram[10] != 1 {
		t.Errorf("A hours[9]=%d hours[10]=%d, want 2 and 1", a.ActiveHoursHistogram[9], a.ActiveHoursHistogram[10])
	}
	if !approxEq(a.MonologueRatio, 1.0/3.0) {
		t.Errorf("A monologue_ratio = %v, want 1/3", a.MonologueRatio)
	}
	if a.ResponseLatencyMedianS != 60 {
		t.Errorf("A latency = %d, want 60", a.ResponseLatencyMedianS)
	}

	b := s.PerSender["userB"]
	if b.Messages != 1 || b.Initiations != 0 {
		t.Errorf("B messages=%d init=%d, want 1 and 0", b.Messages, b.Initiations)
	}
	if b.ResponseLatencyMedianS != 120 {
		t.Errorf("B latency = %d, want 120", b.ResponseLatencyMedianS)
	}

	pw := s.Pairwise
	if pw == nil {
		t.Fatal("pairwise nil, want populated")
	}
	if !approxEq(pw.InitiativeBalance, 1.0) {
		t.Errorf("initiative_balance = %v, want 1.0", pw.InitiativeBalance)
	}
	if !approxEq(pw.Reciprocity, 2.0/3.0) {
		t.Errorf("reciprocity = %v, want 2/3", pw.Reciprocity)
	}
	if pw.AvgReplyGapS != 90 {
		t.Errorf("avg_reply_gap_s = %d, want 90", pw.AvgReplyGapS)
	}
	if len(pw.SilenceGaps) != 1 || pw.SilenceGaps[0].Seconds != 5220 {
		t.Errorf("silence_gaps = %+v, want one gap of 5220s", pw.SilenceGaps)
	}
}

func TestComputeStats_ThreeSendersNoPairwise(t *testing.T) {
	base := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 1, FromID: "userA", Sender: "A", Date: base, ExtractedText: "hi"},
		{ID: 2, FromID: "userB", Sender: "B", Date: base.Add(1 * time.Minute), ExtractedText: "yo"},
		{ID: 3, FromID: "userC", Sender: "C", Date: base.Add(2 * time.Minute), ExtractedText: "hey"},
	}
	s := ComputeStats(msgs, "userA", 30*time.Minute)
	if s.Pairwise != nil {
		t.Fatalf("Pairwise = %+v, want nil for 3 senders", s.Pairwise)
	}
	if len(s.PerSender) != 3 {
		t.Fatalf("per_sender = %d, want 3", len(s.PerSender))
	}
}

func TestComputeStats_EmojiAndTopTerms(t *testing.T) {
	base := time.Date(2026, 1, 2, 9, 0, 0, 0, time.UTC)
	msgs := []dialogue.Message{
		{ID: 1, FromID: "userA", Sender: "A", Date: base, ExtractedText: "работа и работа 🔥"},
		{ID: 2, FromID: "userA", Sender: "A", Date: base.Add(time.Minute), ExtractedText: "снова работа 🔥"},
		{ID: 3, FromID: "userA", Sender: "A", Date: base.Add(2 * time.Minute), MediaType: "sticker", StickerEmoji: "😀"},
	}
	s := ComputeStats(msgs, "userA", 30*time.Minute)
	a := s.PerSender["userA"]
	if a.StickerCount != 1 {
		t.Errorf("sticker_count = %d, want 1", a.StickerCount)
	}
	if len(a.EmojiTop) == 0 || a.EmojiTop[0].Emoji != "🔥" || a.EmojiTop[0].Count != 2 {
		t.Errorf("emoji_top[0] = %+v, want 🔥 x2", a.EmojiTop)
	}
	if len(a.TopTerms) == 0 || a.TopTerms[0].Term != "работа" || a.TopTerms[0].Count != 3 {
		t.Errorf("top_terms[0] = %+v, want работа x3", a.TopTerms)
	}
	for _, tc := range a.TopTerms {
		if tc.Term == "и" {
			t.Errorf("stopword 'и' leaked into top_terms")
		}
	}
}

func TestComputeStats_Empty(t *testing.T) {
	s := ComputeStats(nil, "userA", 30*time.Minute)
	if len(s.PerSender) != 0 || s.Pairwise != nil {
		t.Fatalf("empty input should yield empty stats, got %+v", s)
	}
}
