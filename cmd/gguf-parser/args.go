package main

import (
	"strings"
)

// llama.cpp gives some boolean flags an on|off|auto value, while GGUF Parser models them as
// plain booleans. The two spellings are:
//
//	llama.cpp:    -fa, --flash-attn [on|off|auto]
//	GGUF Parser:  --flash-attention, --flash-attn
//
// Go's flag package stops parsing at the first argument that is not a flag, so "--flash-attn on"
// hides every argument after it: the model source is never read and the command exits with
// "no model specified". "--flash-attn=on" is worse, it prints the help text and exits 0 without
// estimating anything.
//
// Translating the llama.cpp spelling before the flags are parsed keeps both dialects working.
//
// The llama.cpp declarations live in common/arg.cpp.

// llamaValueBoolFlags are the GGUF Parser flags that llama.cpp declares with an on|off|auto value.
var llamaValueBoolFlags = map[string]bool{
	"flash-attention": true,
	"flash-attn":      true,
	"fa":              true,
	"diffusion-fa":    true,
}

// llamaBoolValues maps the value words llama.cpp accepts to the booleans Go's flag package
// accepts.
//
// "auto" means "enable it when the device supports it", and the device decides at runtime. The
// estimate cannot tell whether the device supports it, so "auto" is disabled. The backend turns
// the feature off when the device does not support it, and that state uses more memory.
var llamaBoolValues = map[string]string{
	"on":       "true",
	"enabled":  "true",
	"true":     "true",
	"1":        "true",
	"off":      "false",
	"disabled": "false",
	"false":    "false",
	"0":        "false",
	"auto":     "false",
	"-1":       "false",
}

// normalizeArgs translates the llama.cpp spelling of the boolean flags into the GGUF Parser
// spelling. Every other argument is left alone.
func normalizeArgs(args []string) []string {
	normalized := make([]string, 0, len(args))
	for i := 0; i < len(args); i++ {
		arg := args[i]

		name, value, hasValue := strings.Cut(arg, "=")
		if !llamaValueBoolFlags[strings.TrimLeft(name, "-")] {
			normalized = append(normalized, arg)
			continue
		}

		if hasValue {
			if v, ok := llamaBoolValues[strings.ToLower(value)]; ok {
				normalized = append(normalized, name+"="+v)
				continue
			}
			normalized = append(normalized, arg)
			continue
		}

		// The value is the next argument, as in "--flash-attn on".
		if i+1 < len(args) {
			if v, ok := llamaBoolValues[strings.ToLower(args[i+1])]; ok {
				normalized = append(normalized, name+"="+v)
				i++
				continue
			}
		}

		normalized = append(normalized, arg)
	}
	return normalized
}
