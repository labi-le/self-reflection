package llamacpp

// This file holds the minimal OpenAI-compatible chat-completions wire DTOs the
// persona Brain marshals and decodes. Only the fields the persona bot needs are
// present; the endpoint is served by a llama.cpp server (a later task).

// roleSystem is the role of the injected persona turn that leads every request.
const roleSystem = "system"

// chatMessage is one chat turn: a role ("system"/"user"/"assistant") and its
// text content. It maps 1:1 to a bot.Message plus the leading persona turn.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatRequest is the /v1/chat/completions request body: a non-streaming
// completion of Messages at Temperature for the named Model. RepeatPenalty
// (llama.cpp's own field), PresencePenalty and FrequencyPenalty (OpenAI-standard)
// curb the repetition loop / degeneration a bare temperature leaves unchecked,
// and RepeatLastN widens the penalty look-back so a phrase repeated several turns
// ago (past llama.cpp's default 64-token window) is still penalised; all are
// omitted when zero so the server keeps its own default.
type chatRequest struct {
	Model            string        `json:"model"`
	Stream           bool          `json:"stream"`
	Temperature      float64       `json:"temperature"`
	RepeatPenalty    float64       `json:"repeat_penalty,omitempty"`
	PresencePenalty  float64       `json:"presence_penalty,omitempty"`
	FrequencyPenalty float64       `json:"frequency_penalty,omitempty"`
	RepeatLastN      int           `json:"repeat_last_n,omitempty"`
	Messages         []chatMessage `json:"messages"`
}

// chatResponse is the subset of the response envelope the Brain reads: the first
// choice's assistant message content is the persona's reply.
type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}
