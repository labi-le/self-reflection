package portrait

import (
	"testing"

	"tg2llm/internal/domain/dialogue"
)

func TestChatSummaries(t *testing.T) {
	msgs := []dialogue.Message{
		{ID: 1, ChatID: "200", ChatType: "personal_chat", Sender: "A", FromID: "userA"},
		{ID: 2, ChatID: "200", ChatType: "personal_chat", Sender: "B", FromID: "userB"},
		{ID: 3, ChatID: "200", ChatType: "personal_chat", Sender: "A", FromID: "userA"},
		{ID: 4, ChatID: "300", ChatType: "public_supergroup", Sender: "C", FromID: "userC"},
	}
	cs := ChatSummaries(msgs)
	if len(cs) != 2 {
		t.Fatalf("chats = %d, want 2", len(cs))
	}
	// Ordered by message count desc: chat 200 (3) before 300 (1).
	if cs[0].ChatID != "200" || cs[0].Messages != 3 || cs[0].ChatType != "personal_chat" {
		t.Fatalf("top chat = %+v", cs[0])
	}
	if len(cs[0].Senders) == 0 || cs[0].Senders[0].Term != "A" || cs[0].Senders[0].Count != 2 {
		t.Fatalf("top sender = %+v", cs[0].Senders)
	}
	if cs[1].ChatID != "300" || cs[1].Messages != 1 {
		t.Fatalf("second chat = %+v", cs[1])
	}
}
