package runner

import (
	"context"
	"strconv"
	"strings"

	"tg2llm/internal/bot"
	"tg2llm/pkg/ctxlog"
)

const (
	ownerOnly = "команда только для владельца"
	userHelp  = "/clean — очистить историю\n" +
		"/whoami — узнать свой id"
	ownerHelp = "команды:\n" +
		"/allow <id> [заметка] — разрешить\n" +
		"/deny <id> — запретить\n" +
		"/list — список разрешённых\n" +
		"/clean — очистить историю\n" +
		"/whoami — узнать свой id"
)

// isCommand reports whether text is a slash-command: a leading "/" after
// trimming surrounding whitespace.
func isCommand(text string) bool {
	return strings.HasPrefix(strings.TrimSpace(text), "/")
}

// handleCommand dispatches an owner/utility slash-command and reports whether it
// consumed the update. When it returns true the caller must NOT fall through to
// the persona brain, so a command is never persisted as conversation nor
// answered by the model. Admin commands (allow/deny/list) require the owner;
// whoami/id and help are open to any sender that passed the allow-gate; an
// unknown command is consumed silently so the persona never answers it.
func handleCommand(ctx context.Context, deps Deps, up bot.Update) bool {
	fields := strings.Fields(up.Text)
	if len(fields) == 0 {
		return false
	}
	cmd := strings.ToLower(strings.SplitN(strings.TrimPrefix(fields[0], "/"), "@", 2)[0])
	log := ctxlog.Op(deps.Logger, "runner.handleCommand").With().
		Int64("chat_id", up.ChatID).Int64("user_id", up.UserID).Str("cmd", cmd).Logger()

	reply := func(text string) {
		if err := deps.Transport.Send(ctx, up.ChatID, text); err != nil {
			log.Warn().Err(err).Msg("send command reply failed")
		}
	}

	switch cmd {
	case "whoami", "id":
		reply("твой id: " + strconv.FormatInt(up.UserID, 10))
	case "help", "start":
		if deps.Store.IsOwner(up.UserID) {
			reply(ownerHelp)
		} else {
			reply(userHelp)
		}
	case "allow":
		if !deps.Store.IsOwner(up.UserID) {
			reply(ownerOnly)
			return true
		}
		id, ok := parseID(fields)
		if !ok {
			reply("формат: /allow <id> [заметка]")
			return true
		}
		if err := deps.Store.AddAllowed(id, strings.Join(fields[2:], " ")); err != nil {
			log.Warn().Err(err).Msg("add allowed failed")
		}
		reply("добавил " + strconv.FormatInt(id, 10))
	case "deny":
		if !deps.Store.IsOwner(up.UserID) {
			reply(ownerOnly)
			return true
		}
		id, ok := parseID(fields)
		if !ok {
			reply("формат: /deny <id>")
			return true
		}
		if err := deps.Store.RemoveAllowed(id); err != nil {
			log.Warn().Err(err).Msg("remove allowed failed")
		}
		reply("убрал " + strconv.FormatInt(id, 10))
	case "list":
		if !deps.Store.IsOwner(up.UserID) {
			reply(ownerOnly)
			return true
		}
		entries, err := deps.Store.ListAllowed()
		if err != nil {
			log.Warn().Err(err).Msg("list allowed failed")
		}
		if len(entries) == 0 {
			reply("список пуст")
			return true
		}
		var b strings.Builder
		b.WriteString("разрешены (" + strconv.Itoa(len(entries)) + "):")
		for _, e := range entries {
			b.WriteString("\n" + strconv.FormatInt(e.UserID, 10))
			if e.Note != "" {
				b.WriteString(" — " + e.Note)
			}
		}
		reply(b.String())
	case "clean":
		if err := deps.Store.ClearHistory(up.ChatID); err != nil {
			log.Warn().Err(err).Msg("clear history failed")
		}
		reply("почистил, начнём заново")
	}
	return true
}

// parseID reads the id argument (fields[1]) of a command, reporting ok=false
// when it is missing or not a base-10 int64.
func parseID(fields []string) (int64, bool) {
	if len(fields) < 2 {
		return 0, false
	}
	id, err := strconv.ParseInt(fields[1], 10, 64)
	if err != nil {
		return 0, false
	}
	return id, true
}
