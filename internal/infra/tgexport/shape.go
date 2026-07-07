package tgexport

import (
	"fmt"
	"strings"

	"tg2llm/internal/domain/dialogue"
)

// isFullAccount reports whether the root object is a full-account export (as
// opposed to a single-chat one), detected by any of its account-scoped keys.
func isFullAccount(root map[string]interface{}) bool {
	for _, key := range [...]string{"personal_information", "chats", "left_chats"} {
		if _, ok := root[key]; ok {
			return true
		}
	}
	return false
}

// fullAccountChats returns the chat objects of a full-account export in order:
// chats.list[] followed by left_chats.list[].
func fullAccountChats(root map[string]interface{}) []map[string]interface{} {
	var chats []map[string]interface{}
	for _, key := range [...]string{"chats", "left_chats"} {
		container, ok := root[key].(map[string]interface{})
		if !ok {
			continue
		}
		list, ok := container["list"].([]interface{})
		if !ok {
			continue
		}
		for _, item := range list {
			if chat, ok := item.(map[string]interface{}); ok {
				chats = append(chats, chat)
			}
		}
	}
	return chats
}

// singleChat extracts the messages array and the root chat type/id for the
// legacy shapes: an object with a "messages" field, or a bare array (which
// carries no chat type/id).
func singleChat(raw interface{}) (chatType, chatID string, rawMsgs []interface{}) {
	switch v := raw.(type) {
	case map[string]interface{}:
		if arr, ok := v["messages"].([]interface{}); ok {
			rawMsgs = arr
		}
		chatType = getString(v, "type")
		chatID = numberToString(v["id"])
	case []interface{}:
		rawMsgs = v
	}
	return chatType, chatID, rawMsgs
}

// detectOwnerFullAccount resolves the account owner for a full-account export:
// (1) an explicit hint wins; (2) personal_information.user_id -> "user"+id;
// otherwise it fails loudly, because a full-account export always implies a
// single owning account.
func (r *Reader) detectOwnerFullAccount(root map[string]interface{}) (string, error) {
	if r.ownerHint != "" {
		return r.ownerHint, nil
	}
	if pi, ok := root["personal_information"].(map[string]interface{}); ok {
		if id := numberToString(pi["user_id"]); id != "" {
			return "user" + id, nil
		}
	}
	return "", fmt.Errorf(
		"could not auto-detect the account owner from this full-account export " +
			"(no personal_information.user_id); pass --owner-user-id <id>")
}

// detectOwnerSingleChat resolves the owner for a single-chat export: an explicit
// hint wins; otherwise a personal_chat / bot_chat is inferred as a 1-on-1. Any
// other single chat yields an empty owner (no error) so the plain-text pipeline
// keeps working without identity data.
func (r *Reader) detectOwnerSingleChat(chatType, chatID string, messages []dialogue.Message) string {
	if r.ownerHint != "" {
		return r.ownerHint
	}
	if chatType == "personal_chat" || chatType == "bot_chat" {
		if owner, ok := infer1on1Owner(chatID, messages); ok {
			return owner
		}
	}
	return ""
}

// infer1on1Owner picks the owner of a 1-on-1 chat: it requires exactly two
// distinct user from_ids, one of which is the counterparty ("user"+chatID); the
// other is the owner. Peer-id tails are compared as strings, never as ints.
func infer1on1Owner(chatID string, messages []dialogue.Message) (string, bool) {
	if chatID == "" {
		return "", false
	}
	counterparty := "user" + chatID
	distinct := make(map[string]bool)
	for _, m := range messages {
		if strings.HasPrefix(m.FromID, "user") {
			distinct[m.FromID] = true
		}
	}
	if len(distinct) != 2 || !distinct[counterparty] {
		return "", false
	}
	for id := range distinct {
		if id != counterparty {
			return id, true
		}
	}
	return "", false
}
