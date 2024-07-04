package stringx

import "strings"

// CutFromLeft is the same as strings.Cut,
// which starts from left to right,
// slices s around the first instance of sep,
// returning the text before and after sep.
// The found result reports whether sep appears in s.
// If sep does not appear in s, cut returns s, "", false.
func CutFromLeft(s, sep string) (before, after string, found bool) {
	return strings.Cut(s, sep)
}

// CutFromRight takes the same arguments as CutFromLeft,
// but starts from right to left,
// slices s around the last instance of sep,
// return the text before and after sep.
// The found result reports whether sep appears in s.
// If sep does not appear in s, cut returns s, "", false.
func CutFromRight(s, sep string) (before, after string, found bool) {
	if i := strings.LastIndex(s, sep); i >= 0 {
		return s[:i], s[i+len(sep):], true
	}
	return s, "", false
}
