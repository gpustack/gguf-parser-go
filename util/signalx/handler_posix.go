//go:build !windows

package signalx

import (
	"os"
	"syscall"
)

var sigs = []os.Signal{syscall.SIGINT, syscall.SIGTERM}
