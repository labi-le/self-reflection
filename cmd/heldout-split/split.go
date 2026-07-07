package main

import (
	"math/rand"
	"sort"
)

// splitExamples returns the train and held-out index partitions. Selection is
// deterministic for a given seed, drawn only from assistant-bearing examples,
// and stratified across (chatID, bucket) via seeded per-stratum shuffles picked
// round-robin. If heldout exceeds the eligible count it is capped at
// floor(0.1*total) and warn is called. Both slices are in original input order.
func splitExamples(exs []example, seed int64, heldout int, warn func(string, ...any)) (train, held []int) {
	total := len(exs)

	eligible := 0
	byStratum := make(map[string][]int)
	for _, e := range exs {
		if !e.hasAssistant {
			continue
		}
		eligible++
		key := e.chatID + "\x00" + e.bucket
		byStratum[key] = append(byStratum[key], e.index)
	}

	desired := heldout
	if desired < 0 {
		desired = 0
	}
	if desired > eligible {
		capped := total / 10 // floor(0.1*total)
		if capped < desired {
			desired = capped
		}
		if warn != nil {
			warn("requested heldout=%d exceeds eligible=%d; capping to %d (floor(0.1*total), total=%d)",
				heldout, eligible, desired, total)
		}
	}
	if desired > eligible {
		desired = eligible
	}

	keys := make([]string, 0, len(byStratum))
	for k := range byStratum {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	rng := rand.New(rand.NewSource(seed))
	for _, k := range keys {
		idxs := byStratum[k]
		rng.Shuffle(len(idxs), func(i, j int) { idxs[i], idxs[j] = idxs[j], idxs[i] })
	}

	heldSet := make(map[int]bool, desired)
	for len(heldSet) < desired {
		progressed := false
		for _, k := range keys {
			if len(heldSet) >= desired {
				break
			}
			idxs := byStratum[k]
			if len(idxs) == 0 {
				continue
			}
			heldSet[idxs[0]] = true
			byStratum[k] = idxs[1:]
			progressed = true
		}
		if !progressed {
			break
		}
	}

	for _, e := range exs {
		if heldSet[e.index] {
			held = append(held, e.index)
		} else {
			train = append(train, e.index)
		}
	}
	return train, held
}
