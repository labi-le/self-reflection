package botapi

import (
	"context"
	"strings"
	"testing"
)

// The bot token lives in every request path, so a naive error (or a log of one)
// on a transport failure would leak it. This locks the redaction contract: a
// connection failure surfaces a *url.Error carrying the token, and it must reach
// the caller through the public Send API with the token masked — never in clear.
func Test_Send_never_leaks_the_bot_token_in_errors(t *testing.T) {
	// Given a token and a base URL that refuses connections
	const token = "1234567:SUPERSECRET-abcXYZ"
	c := New(token, allowAll, testLogger, WithBaseURL("http://127.0.0.1:1"), withSleep(immediateSleep))

	// When Send attempts delivery and the transport fails
	err := c.Send(context.Background(), 100, "hello")

	// Then the error surfaces, but the raw token never appears in it
	if err == nil {
		t.Fatal("expected a transport error from Send, got nil")
	}
	if strings.Contains(err.Error(), token) {
		t.Fatalf("bot token leaked into Send error: %v", err)
	}
	if !strings.Contains(err.Error(), tokenRedaction) {
		t.Fatalf("expected the redaction marker in the error (proves the failing path ran), got: %v", err)
	}
}
