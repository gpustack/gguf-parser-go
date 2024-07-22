package signalx

import (
	"context"
	"os"
	"os/signal"
)

var registered = make(chan struct{})

// Handler registers for signals and returns a context.
func Handler() context.Context {
	close(registered) // Panics when called twice.

	sigChan := make(chan os.Signal, len(sigs))
	ctx, cancel := context.WithCancel(context.Background())

	// Register for signals.
	signal.Notify(sigChan, sigs...)

	// Process signals.
	go func() {
		var exited bool
		for range sigChan {
			if exited {
				os.Exit(1)
			}
			cancel()
			exited = true
		}
	}()

	return ctx
}
