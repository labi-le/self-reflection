package media

import "testing"

func TestOneLine(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"collapses newlines", "a\nb\nc", "a b c"},
		{"trims each line", "  a  \n  b  ", "a b"},
		{"drops empty lines", "a\n\n\nb", "a b"},
		{"strips carriage returns", "a\r\nb\r\n", "a b"},
		{"all blank", "  \n\t\n  ", ""},
		{"single line", "hello", "hello"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := OneLine(c.in); got != c.want {
				t.Fatalf("OneLine(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}
