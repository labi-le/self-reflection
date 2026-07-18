package main

import (
	"flag"
	"fmt"
	"time"

	"tg2llm/internal/domain/portrait"
)

// runParticipants prints the senders in an export (id, display name, message
// count) so a user can discover a valid --target/--with for portrait/ask. No
// LLM involved — it is just the deterministic stats pass.
func runParticipants(args []string) error {
	fs := flag.NewFlagSet("participants", flag.ExitOnError)
	sessionGap := fs.Duration("session-gap", 30*time.Minute, "silence gap that starts a new session")
	verbose := fs.Bool("verbose", false, "trace logging")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if fs.NArg() < 1 {
		return fmt.Errorf("usage: analyze participants [flags] <export.json>")
	}
	logger := initLogger(*verbose)
	msgs, _, err := readMessages(fs.Arg(0), logger)
	if err != nil {
		return err
	}
	stats := portrait.ComputeStats(msgs, "", *sessionGap)
	fmt.Printf("%-26s  %-24s  %s\n", "FROM_ID", "DISPLAY_NAME", "MESSAGES")
	for _, id := range sortedSenders(stats) {
		s := stats.PerSender[id]
		fmt.Printf("%-26s  %-24s  %d\n", id, s.DisplayName, s.Messages)
	}
	return nil
}
