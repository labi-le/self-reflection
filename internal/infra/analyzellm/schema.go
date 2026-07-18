package analyzellm

import (
	"encoding/json"

	"tg2llm/internal/domain/portrait"
)

// fragmentSchema is the OpenAI response_format json_schema for the flat M1
// fragment. It is built in Go from the domain catalogs (portrait.OCEAN, the
// Confidence constants) so the schema and the prompt cannot drift. Every object
// forbids extra keys so the model returns exactly the shape guard() expects.
func fragmentSchema() json.RawMessage {
	trait := object(
		props{
			"score":      map[string]any{"type": []string{"number", "null"}},
			"confidence": confidenceEnum(),
			"evidence":   evidenceArray(),
		},
		"score", "confidence", "evidence",
	)

	bigFiveProps := props{}
	bigFiveRequired := make([]string, 0, len(portrait.OCEAN))
	for _, dim := range portrait.OCEAN {
		bigFiveProps[dim] = trait
		bigFiveRequired = append(bigFiveRequired, dim)
	}

	commStyle := object(
		props{
			"tone":                     stringType(),
			"formality":                stringType(),
			"directness":               stringType(),
			"humor":                    stringType(),
			"emotional_expressiveness": stringType(),
			"conflict_style":           stringType(),
			"confidence":               confidenceEnum(),
			"evidence":                 evidenceArray(),
		},
		"tone", "formality", "directness", "humor",
		"emotional_expressiveness", "conflict_style", "confidence", "evidence",
	)

	topic := object(
		props{
			"topic":      stringType(),
			"weight":     map[string]any{"type": "number"},
			"confidence": confidenceEnum(),
			"evidence":   evidenceArray(),
		},
		"topic", "weight", "confidence", "evidence",
	)

	claim := claimObject()

	accentuation := object(
		props{
			"type":       enumType(portrait.AccentuationTypes),
			"intensity":  map[string]any{"type": "number"},
			"confidence": confidenceEnum(),
			"evidence":   evidenceArray(),
		},
		"type", "intensity", "confidence", "evidence",
	)

	emotionalPattern := object(
		props{
			"pattern":    stringType(),
			"triggers":   arrayOf(stringType()),
			"confidence": confidenceEnum(),
			"evidence":   evidenceArray(),
		},
		"pattern", "triggers", "confidence", "evidence",
	)

	prediction := object(
		props{
			"prediction": stringType(),
			"confidence": confidenceEnum(),
			"evidence":   evidenceArray(),
		},
		"prediction", "confidence", "evidence",
	)

	schema := object(
		props{
			"big_five":            object(bigFiveProps, bigFiveRequired...),
			"accentuations":       arrayOf(accentuation),
			"communication_style": commStyle,
			"recurring_topics":    arrayOf(topic),
			"values_motivations":  arrayOf(claim),
			"emotional_patterns":  arrayOf(emotionalPattern),
			"what_to_expect":      arrayOf(prediction),
		},
		"big_five", "accentuations", "communication_style", "recurring_topics",
		"values_motivations", "emotional_patterns", "what_to_expect",
	)

	format := map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "portrait_fragment",
			"strict": true,
			"schema": schema,
		},
	}
	raw, err := json.Marshal(format)
	if err != nil {
		// The schema is built from static literals; a marshal failure is a
		// programming error, not a runtime condition.
		panic("analyzellm: build fragment schema: " + err.Error())
	}
	return raw
}

// dynamicsSchema is the response_format json_schema for the 1:1 relationship
// reply. The karpman_roles object is keyed by the exact participant identifiers
// a and b so the strict schema (additionalProperties:false) still fits a map
// keyed by participant. Role values are constrained to the domain catalog.
func dynamicsSchema(a, b string) json.RawMessage {
	karpmanRole := object(
		props{
			"role":       enumType(portrait.KarpmanRoles),
			"tendency":   map[string]any{"type": "number"},
			"confidence": confidenceEnum(),
			"evidence":   evidenceArray(),
		},
		"role", "tendency", "confidence", "evidence",
	)

	schema := object(
		props{
			"karpman_roles": object(props{a: arrayOf(karpmanRole), b: arrayOf(karpmanRole)}, a, b),
			"power_balance": claimObject(),
		},
		"karpman_roles", "power_balance",
	)

	format := map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":   "portrait_dynamics",
			"strict": true,
			"schema": schema,
		},
	}
	raw, err := json.Marshal(format)
	if err != nil {
		panic("analyzellm: build dynamics schema: " + err.Error())
	}
	return raw
}

func claimObject() map[string]any {
	return object(
		props{
			"claim":      stringType(),
			"confidence": confidenceEnum(),
			"evidence":   evidenceArray(),
		},
		"claim", "confidence", "evidence",
	)
}

type props = map[string]any

func object(p props, required ...string) map[string]any {
	return map[string]any{
		"type":                 "object",
		"additionalProperties": false,
		"properties":           p,
		"required":             required,
	}
}

func arrayOf(item map[string]any) map[string]any {
	return map[string]any{"type": "array", "items": item}
}

func stringType() map[string]any { return map[string]any{"type": "string"} }

func evidenceArray() map[string]any {
	return map[string]any{"type": "array", "items": map[string]any{"type": "integer"}}
}

func confidenceEnum() map[string]any { return enumType(confidenceValues()) }

func enumType(values []string) map[string]any {
	return map[string]any{"type": "string", "enum": values}
}

func confidenceValues() []string {
	return []string{
		string(portrait.ConfLow),
		string(portrait.ConfMedium),
		string(portrait.ConfHigh),
	}
}
