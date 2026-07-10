package llamacpp

import (
	"context"
	"net/http/httptest"
	"testing"
	"time"

	"tg2llm/internal/bot"
)

// The persona request must carry the empirically-tuned sampling controls that
// stop the model looping and degenerating: temperature 0.3, repeat_penalty 1.1
// and presence_penalty 0.3 by default, each overridable via its With* option.
// This is the serve-side half of the web-UI fix (temp 0.3 + repeat 1.1 +
// presence 0.3); without the penalties the brain repeats one short reply.
func Test_Reply_request_carries_sampling_controls(t *testing.T) {
	tests := []struct {
		name                 string
		opts                 []Option
		wantTemperature      float64
		wantRepeatPenalty    float64
		wantPresencePenalty  float64
		wantFrequencyPenalty float64
		wantRepeatLastN      int
	}{
		{
			name:                 "defaults",
			opts:                 nil,
			wantTemperature:      0.3,
			wantRepeatPenalty:    1.1,
			wantPresencePenalty:  0.3,
			wantFrequencyPenalty: 0.0,
			wantRepeatLastN:      0,
		},
		{
			name:                 "WithTemperature overrides only temperature",
			opts:                 []Option{WithTemperature(0.9)},
			wantTemperature:      0.9,
			wantRepeatPenalty:    1.1,
			wantPresencePenalty:  0.3,
			wantFrequencyPenalty: 0.0,
			wantRepeatLastN:      0,
		},
		{
			name:                 "WithRepeatPenalty overrides only repeat_penalty",
			opts:                 []Option{WithRepeatPenalty(1.5)},
			wantTemperature:      0.3,
			wantRepeatPenalty:    1.5,
			wantPresencePenalty:  0.3,
			wantFrequencyPenalty: 0.0,
			wantRepeatLastN:      0,
		},
		{
			name:                 "WithPresencePenalty overrides only presence_penalty",
			opts:                 []Option{WithPresencePenalty(0.7)},
			wantTemperature:      0.3,
			wantRepeatPenalty:    1.1,
			wantPresencePenalty:  0.7,
			wantFrequencyPenalty: 0.0,
			wantRepeatLastN:      0,
		},
		{
			name:                 "WithFrequencyPenalty overrides only frequency_penalty",
			opts:                 []Option{WithFrequencyPenalty(0.6)},
			wantTemperature:      0.3,
			wantRepeatPenalty:    1.1,
			wantPresencePenalty:  0.3,
			wantFrequencyPenalty: 0.6,
			wantRepeatLastN:      0,
		},
		{
			name:                 "WithRepeatLastN overrides only repeat_last_n",
			opts:                 []Option{WithRepeatLastN(384)},
			wantTemperature:      0.3,
			wantRepeatPenalty:    1.1,
			wantPresencePenalty:  0.3,
			wantFrequencyPenalty: 0.0,
			wantRepeatLastN:      384,
		},
		{
			name:                 "all five overridden together",
			opts:                 []Option{WithTemperature(0.5), WithRepeatPenalty(1.2), WithPresencePenalty(0.4), WithFrequencyPenalty(0.6), WithRepeatLastN(256)},
			wantTemperature:      0.5,
			wantRepeatPenalty:    1.2,
			wantPresencePenalty:  0.4,
			wantFrequencyPenalty: 0.6,
			wantRepeatLastN:      256,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Given a persona server that records the request body
			f := &fakePersona{reply: "ок"}
			srv := httptest.NewServer(f)
			defer srv.Close()
			opts := append([]Option{WithTimeout(2 * time.Second)}, tt.opts...)
			c := New(srv.URL, "", "", opts...)

			// When one reply is generated
			if _, err := c.Reply(context.Background(), []bot.Message{{Role: "user", Content: "привет"}}); err != nil {
				t.Fatalf("Reply: %v", err)
			}

			// Then the wire request carries the expected sampling controls
			got := f.lastRequest(t)
			if got.Temperature != tt.wantTemperature {
				t.Errorf("temperature = %v, want %v", got.Temperature, tt.wantTemperature)
			}
			if got.RepeatPenalty != tt.wantRepeatPenalty {
				t.Errorf("repeat_penalty = %v, want %v", got.RepeatPenalty, tt.wantRepeatPenalty)
			}
			if got.PresencePenalty != tt.wantPresencePenalty {
				t.Errorf("presence_penalty = %v, want %v", got.PresencePenalty, tt.wantPresencePenalty)
			}
			if got.FrequencyPenalty != tt.wantFrequencyPenalty {
				t.Errorf("frequency_penalty = %v, want %v", got.FrequencyPenalty, tt.wantFrequencyPenalty)
			}
			if got.RepeatLastN != tt.wantRepeatLastN {
				t.Errorf("repeat_last_n = %v, want %v", got.RepeatLastN, tt.wantRepeatLastN)
			}
		})
	}
}
