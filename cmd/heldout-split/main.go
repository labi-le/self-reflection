// Command heldout-split deterministically partitions a ChatML/"messages" JSONL
// dataset into a train set and a held-out eval set for a style-fidelity
// perplexity harness.
//
// Each input line is one example: {"messages":[{"role":"...","content":"..."},...]}.
// Only examples that contain an owner assistant reply (a message with role
// "assistant") are eligible to be held out. Selection is deterministic given a
// seed and stratified across source chat id + a coarse (year-month) time bucket,
// so the eval set stays representative and reruns are byte-identical.
//
// Convention (Go's flag parser): all --flags come before any positional args.
// This tool takes no positionals.
package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
)

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run(args []string) error {
	fs := flag.NewFlagSet("heldout-split", flag.ContinueOnError)
	fs.Usage = func() {
		fmt.Fprintf(fs.Output(), "Usage: heldout-split [flags]\n\nFlags:\n")
		fs.PrintDefaults()
	}
	in := fs.String("in", "", "input JSONL dataset ({\"messages\":[...]} per line) (required)")
	trainOut := fs.String("train-out", "data/train.jsonl", "output path for the train split")
	heldoutOut := fs.String("heldout-out", "data/heldout.jsonl", "output path for the held-out eval split")
	seed := fs.Int64("seed", 42, "deterministic RNG seed")
	heldout := fs.Int("heldout", 2000, "approximate number of held-out eval examples")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *in == "" {
		return errors.New("--in is required")
	}

	f, err := os.Open(*in)
	if err != nil {
		return fmt.Errorf("opening --in: %w", err)
	}
	defer f.Close()

	examples, err := parseExamples(f)
	if err != nil {
		return err
	}

	warn := func(format string, a ...any) {
		fmt.Fprintf(os.Stderr, "warning: "+format+"\n", a...)
	}
	train, held := splitExamples(examples, *seed, *heldout, warn)

	if err := writeSplit(*trainOut, examples, train); err != nil {
		return err
	}
	if err := writeSplit(*heldoutOut, examples, held); err != nil {
		return err
	}

	fmt.Printf("split %d examples (seed=%d): train=%d heldout=%d -> %s, %s\n",
		len(examples), *seed, len(train), len(held), *trainOut, *heldoutOut)
	return nil
}

// writeSplit writes the selected examples' verbatim lines, in index order, one
// per line, creating the parent directory as needed.
func writeSplit(path string, exs []example, idxs []int) error {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return fmt.Errorf("creating dir for %s: %w", path, err)
		}
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("creating %s: %w", path, err)
	}
	defer f.Close()

	w := bufio.NewWriter(f)
	for _, i := range idxs {
		if _, err := w.WriteString(exs[i].raw); err != nil {
			return err
		}
		if err := w.WriteByte('\n'); err != nil {
			return err
		}
	}
	return w.Flush()
}
