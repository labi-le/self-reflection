package main

import (
	"flag"
	"fmt"
	"strings"

	"tg2llm/internal/domain/portrait"
)

// runChats lists the chats in an export (chat_id, type, message count, top
// senders) so a user can pick a chat_id for portrait/ask --chat. No LLM — it is
// the deterministic per-chat grouping.
func runChats(args []string) error {
	fs := flag.NewFlagSet("chats", flag.ExitOnError)
	verbose := fs.Bool("verbose", false, "trace logging")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: analyze chats [flags] <export.json>")
	}
	logger := initLogger(*verbose)
	msgs, _, err := readMessages(fs.Arg(0), logger)
	if err != nil {
		return err
	}
	fmt.Printf("%-16s  %-20s  %-8s  %s\n", "CHAT_ID", "TYPE", "MESSAGES", "TOP SENDERS")
	for _, c := range portrait.ChatSummaries(msgs) {
		names := make([]string, 0, len(c.Senders))
		for _, s := range c.Senders {
			names = append(names, fmt.Sprintf("%s(%d)", s.Term, s.Count))
		}
		fmt.Printf("%-16s  %-20s  %-8d  %s\n", c.ChatID, c.ChatType, c.Messages, strings.Join(names, ", "))
	}
	return nil
}
