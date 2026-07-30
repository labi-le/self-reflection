package main

import (
	"fmt"
	"os"

	"tg2llm/internal/app"
	"tg2llm/internal/domain/dialogue"
	"tg2llm/internal/infra/chatmlsink"
	"tg2llm/internal/infra/textsink"
	"tg2llm/internal/infra/tgexport"

	"github.com/rs/zerolog"
)

// cachedMessages is a MessageSource backed by an already-parsed export. ChatML
// mode reads the export once in buildPipeline to resolve the account owner, then
// replays those messages here so the service does not parse the (potentially
// large) file a second time.
type cachedMessages struct {
	messages []dialogue.Message
	refs     map[int]bool
}

func (c cachedMessages) ReadMessages() ([]dialogue.Message, map[int]bool, error) {
	return c.messages, c.refs, nil
}

// buildPipeline selects the message source and dialogue sink for the chosen
// output format. Text mode (the default) is unchanged: the reader streams
// straight into a textsink. ChatML mode needs the account owner up front because
// the sink's role-mapping keys off it, so it reads the export ONCE here to
// resolve the owner (an explicit --owner-user-id wins, otherwise the reader's
// auto-detected owner) and hands the parsed messages to the service via a cached
// source, avoiding a second parse. The resolved owner is written back into opts:
// the service's media fold keys off the same id, and a disagreement would render
// the owner's decoded media as if someone else had sent it. The returned *os.File
// is non-nil only for chatml — its sink writes to but does not close it — and the
// caller closes it after the run.
func buildPipeline(opts *app.ParseOptions, reader *tgexport.Reader, outputPath string, logger zerolog.Logger) (app.MessageSource, app.DialogueSink, *os.File, error) {
	if opts.Format != "chatml" {
		sink, err := textsink.NewWriter(outputPath)
		if err != nil {
			return nil, nil, nil, err
		}
		return reader, sink, nil, nil
	}

	reader.WithOwnerHint(opts.OwnerFromID)
	messages, refs, err := reader.ReadMessages()
	if err != nil {
		return nil, nil, nil, err
	}
	if opts.OwnerFromID == "" {
		opts.OwnerFromID = reader.OwnerFromID()
	}
	owner := opts.OwnerFromID

	f, err := os.Create(outputPath)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("opening output %s: %w", outputPath, err)
	}
	sink := chatmlsink.NewSink(f, chatmlsink.Options{
		OwnerFromID:   owner,
		PersonaSystem: opts.PersonaSystem,
		BurstDelim:    opts.BurstDelim,
		SessionGap:    opts.SessionGap,
		CtxTurns:      opts.CtxTurns,
		CutoffLen:     opts.CutoffLen,
		MaxPerChat:    opts.MaxPerChat,
		Logger:        logger,
	})
	return cachedMessages{messages: messages, refs: refs}, sink, f, nil
}
