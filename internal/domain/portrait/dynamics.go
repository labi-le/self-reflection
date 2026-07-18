package portrait

import (
	"sort"
	"strconv"
	"time"

	"tg2llm/internal/domain/dialogue"
)

// arcEvidenceCap bounds how many message ids one relationship-arc window cites,
// so a chatty month does not bloat the evidence index.
const arcEvidenceCap = 20

// ReduceDynamics merges per-chunk relationship fragments into one Dynamics using
// the same deterministic, evidence-weighted aggregation as reduce.go: each
// participant's Karpman roles are grouped by role name and merged as an
// evidence-count-weighted tendency mean (evidence unioned+sorted, confidence
// maxed); power balance is the highest-evidence non-empty claim with its
// evidence unioned across chunks. Non-catalog roles and any item left with empty
// evidence are dropped. RelationshipArc is set separately by the caller.
func ReduceDynamics(a, b string, dyns []Dynamics) Dynamics {
	return Dynamics{
		KarpmanRoles: reduceKarpman(dyns),
		PowerBalance: reducePowerBalance(dyns),
	}
}

func reduceKarpman(dyns []Dynamics) map[string][]KarpmanRole {
	// participant id -> normalized role name -> aggregate.
	groups := map[string]map[string]*numAgg{}
	for _, d := range dyns {
		for who, roles := range d.KarpmanRoles {
			for _, r := range roles {
				if !isKarpmanRole(r.Role) {
					continue
				}
				pg := groups[who]
				if pg == nil {
					pg = map[string]*numAgg{}
					groups[who] = pg
				}
				k := normKey(r.Role)
				g := pg[k]
				if g == nil {
					g = &numAgg{key: k, order: len(pg), conf: ConfLow}
					pg[k] = g
				}
				g.add(r.Tendency, r.Evidence, r.Confidence)
			}
		}
	}
	if len(groups) == 0 {
		return nil
	}
	out := map[string][]KarpmanRole{}
	for who, pg := range groups {
		var roles []KarpmanRole
		for _, g := range rankNum(pg) {
			ev := unionSorted(g.ev)
			if len(ev) == 0 {
				continue
			}
			roles = append(roles, KarpmanRole{Role: g.key, Tendency: g.mean(), Confidence: g.conf, Evidence: ev})
		}
		if len(roles) > 0 {
			out[who] = roles
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func reducePowerBalance(dyns []Dynamics) Claim {
	var bestClaim string
	bestEv := -1
	conf := ConfLow
	var ev []int
	for _, d := range dyns {
		pb := d.PowerBalance
		if pb.Claim == "" || len(pb.Evidence) == 0 {
			continue
		}
		ev = append(ev, pb.Evidence...)
		conf = maxConf(conf, pb.Confidence)
		// Highest evidence count wins; ties broken lexicographically so the
		// result is independent of chunk order.
		n := len(pb.Evidence)
		if n > bestEv || (n == bestEv && pb.Claim < bestClaim) {
			bestEv = n
			bestClaim = pb.Claim
		}
	}
	if bestEv < 0 {
		return Claim{}
	}
	return Claim{Claim: bestClaim, Confidence: conf, Evidence: unionSorted(ev)}
}

func isKarpmanRole(role string) bool {
	k := normKey(role)
	for _, r := range KarpmanRoles {
		if r == k {
			return true
		}
	}
	return false
}

// RelationshipArc buckets messages by calendar month (YYYY-MM) and emits one
// deterministic, LLM-free window per month: a factual activity summary (message
// count and how many sessions target initiated), the window's message ids as
// evidence (capped), and ConfLow confidence. This is an honest activity arc, not
// an inferred narrative. Windows are returned in ascending month order.
func RelationshipArc(msgs []dialogue.Message, target string, sessionGap time.Duration) []ArcWindow {
	sorted := make([]dialogue.Message, len(msgs))
	copy(sorted, msgs)
	sort.Slice(sorted, func(i, j int) bool {
		if !sorted[i].Date.Equal(sorted[j].Date) {
			return sorted[i].Date.Before(sorted[j].Date)
		}
		return sorted[i].ID < sorted[j].ID
	})

	type bucket struct {
		n     int
		inits int
		ids   []int
	}
	windows := map[string]*bucket{}
	var prev time.Time
	havePrev := false
	for _, m := range sorted {
		newSession := !havePrev || m.Date.Sub(prev) > sessionGap
		prev = m.Date
		havePrev = true

		key := m.Date.Format("2006-01")
		bkt := windows[key]
		if bkt == nil {
			bkt = &bucket{}
			windows[key] = bkt
		}
		bkt.n++
		if len(bkt.ids) < arcEvidenceCap {
			bkt.ids = append(bkt.ids, m.ID)
		}
		if newSession && m.FromID == target {
			bkt.inits++
		}
	}

	keys := make([]string, 0, len(windows))
	for k := range windows {
		keys = append(keys, k)
	}
	sort.Strings(keys)

	var out []ArcWindow
	for _, k := range keys {
		bkt := windows[k]
		summary := strconv.Itoa(bkt.n) + " messages, " + strconv.Itoa(bkt.inits) + " initiations by " + target
		out = append(out, ArcWindow{
			Window:     k,
			Summary:    summary,
			Confidence: ConfLow,
			Evidence:   bkt.ids,
		})
	}
	return out
}
