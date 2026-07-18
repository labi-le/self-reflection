package analyze

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"tg2llm/internal/domain/portrait"
	"tg2llm/internal/infra/llm"
)

// QARefusal is the fixed answer returned whenever the retrieved excerpts do not
// ground the question: no retrieval, or the model cited nothing that survives
// the member-id guard. It is deliberately a constant so callers can detect it.
const QARefusal = "I can't answer that from the analyzed conversation."

var citationRe = regexp.MustCompile(`\[(\d+)\]`)

// QAService answers questions strictly from the indexed conversation: it
// retrieves the top-k chunks, prompts the model to answer only from those
// excerpts, then discards any cited id absent from the retrieved chunks' member
// ids. An answer with no surviving citation is refused.
type QAService struct {
	chat       *llm.Client
	embedder   Embedder
	index      Index
	collection string
}

// NewQAService wires the chat client, embedder, index, and the per-analysis
// collection name the retrieval queries against.
func NewQAService(chat *llm.Client, e Embedder, idx Index, collection string) *QAService {
	return &QAService{chat: chat, embedder: e, index: idx, collection: collection}
}

// Ask embeds the question, retrieves grounding excerpts, and returns a grounded
// answer with the message ids it cites. Cited ids are intersected with the
// union of retrieved chunks' member ids; if nothing supports the answer (or
// retrieval is empty) it returns QARefusal and nil ids without fabricating.
func (q *QAService) Ask(ctx context.Context, question string, k int) (string, []int, error) {
	vecs, err := q.embedder.Embed(ctx, []string{question})
	if err != nil {
		return "", nil, fmt.Errorf("embed question: %w", err)
	}
	if len(vecs) == 0 {
		return "", nil, fmt.Errorf("embedder returned no vector")
	}
	chunks, err := q.index.Query(ctx, q.collection, vecs[0], k, nil)
	if err != nil {
		return "", nil, fmt.Errorf("query index: %w", err)
	}
	if len(chunks) == 0 {
		return QARefusal, nil, nil
	}

	allowed := map[int]bool{}
	for _, c := range chunks {
		for id := range c.MemberIDs {
			allowed[id] = true
		}
	}

	answer, err := q.chat.Chat(ctx, buildQAPrompt(question, chunks), llm.ChatOptions{Temperature: 0})
	if err != nil {
		return "", nil, fmt.Errorf("chat: %w", err)
	}

	cited := groundedCitations(answer, allowed)
	if len(cited) == 0 {
		return QARefusal, nil, nil
	}
	return answer, cited, nil
}

// buildQAPrompt embeds the retrieved excerpts (each line already carries its
// message id as [<id>]) and instructs the model to answer only from them,
// citing supporting ids, or to refuse verbatim.
func buildQAPrompt(question string, chunks []portrait.Chunk) string {
	var b strings.Builder
	b.WriteString("Answer the question using ONLY the conversation excerpts below.\n")
	b.WriteString("Cite every supporting message by its id in square brackets, e.g. [42].\n")
	b.WriteString("If the excerpts do not contain the answer, reply exactly: ")
	b.WriteString(QARefusal)
	b.WriteString("\n\nExcerpts:\n")
	for _, c := range chunks {
		b.WriteString(c.Text)
		b.WriteString("\n")
	}
	b.WriteString("\nQuestion: ")
	b.WriteString(question)
	b.WriteString("\nAnswer:")
	return b.String()
}

// groundedCitations extracts bracketed message ids from the answer and keeps
// only those present in the allowed set, de-duplicated in first-seen order.
func groundedCitations(answer string, allowed map[int]bool) []int {
	var out []int
	seen := map[int]bool{}
	for _, m := range citationRe.FindAllStringSubmatch(answer, -1) {
		id, err := strconv.Atoi(m[1])
		if err != nil || seen[id] || !allowed[id] {
			continue
		}
		seen[id] = true
		out = append(out, id)
	}
	return out
}
