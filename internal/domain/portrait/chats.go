package portrait

import (
	"sort"

	"tg2llm/internal/domain/dialogue"
)

// ChatSummary describes one chat in an export: enough to pick a --chat/--target
// without the chat title (which the reader does not attach to a Message), the
// top senders identify a 1:1 or a group.
type ChatSummary struct {
	ChatID   string      `json:"chat_id"`
	ChatType string      `json:"chat_type"`
	Messages int         `json:"messages"`
	Senders  []TermCount `json:"senders"` // display name -> message count, most active first
}

// ChatSummaries groups messages by ChatID and returns one summary per chat,
// ordered by message count descending. Pure: no I/O. Messages without a ChatID
// are grouped under the empty key (legacy single-chat exports).
func ChatSummaries(msgs []dialogue.Message) []ChatSummary {
	type acc struct {
		chatType string
		total    int
		senders  map[string]int
	}
	byChat := map[string]*acc{}
	order := []string{}
	for _, m := range msgs {
		a := byChat[m.ChatID]
		if a == nil {
			a = &acc{chatType: m.ChatType, senders: map[string]int{}}
			byChat[m.ChatID] = a
			order = append(order, m.ChatID)
		}
		a.total++
		name := m.Sender
		if name == "" {
			name = m.FromID
		}
		a.senders[name]++
	}

	out := make([]ChatSummary, 0, len(byChat))
	for _, id := range order {
		a := byChat[id]
		senders := make([]TermCount, 0, len(a.senders))
		for n, c := range a.senders {
			senders = append(senders, TermCount{Term: n, Count: c})
		}
		sort.Slice(senders, func(i, j int) bool {
			if senders[i].Count != senders[j].Count {
				return senders[i].Count > senders[j].Count
			}
			return senders[i].Term < senders[j].Term
		})
		if len(senders) > 5 {
			senders = senders[:5]
		}
		out = append(out, ChatSummary{ChatID: id, ChatType: a.chatType, Messages: a.total, Senders: senders})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Messages != out[j].Messages {
			return out[i].Messages > out[j].Messages
		}
		return out[i].ChatID < out[j].ChatID
	})
	return out
}
