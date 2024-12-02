package gguf_parser

import "math"

// GuessSD1DiffusionModelMemoryUsage returns the memory usage in bytes for the given width and height,
// which is calculated by linear regression or polynomial regression.
func GuessSD1DiffusionModelMemoryUsage(width, height uint32, flashAttention bool) uint64 {
	coefficients := []float64{7.8763685671743e+06, 161.42301986333496, 0.007812489338703485}
	degree := 2
	x := float64(width * height)

	y := float64(0)
	for i := 0; i <= degree; i++ {
		y += coefficients[i] * math.Pow(x, float64(i))
	}
	return uint64(y)
}

// GuessSD2DiffusionModelMemoryUsage returns the memory usage in bytes for the given width and height,
// which is calculated by linear regression or polynomial regression.
func GuessSD2DiffusionModelMemoryUsage(width, height uint32, flashAttention bool) uint64 {
	coefficients := []float64{-3.5504397905618614e+08, -1193.3271458642232, 0.005402381760522009}
	degree := 2
	x := float64(width * height)

	if flashAttention {
		coefficients = []float64{3.78068128077788e+06, 513.2102510934714}
		degree = 1
	}

	y := float64(0)
	for i := 0; i <= degree; i++ {
		y += coefficients[i] * math.Pow(x, float64(i))
	}
	return uint64(y)
}

// GuessSDXLDiffusionModelMemoryUsage returns the memory usage in bytes for the given width and height,
// which is calculated by linear regression or polynomial regression.
func GuessSDXLDiffusionModelMemoryUsage(width, height uint32, flashAttention bool) uint64 {
	coefficients := []float64{5.554129038929968e+07, 138.31961166554433, 0.0006109454572342757}
	degree := 2
	x := float64(width * height)

	if flashAttention {
		coefficients = []float64{-5.95880278052181e+06, 500.0687898914631}
		degree = 1
	}

	y := float64(0)
	for i := 0; i <= degree; i++ {
		y += coefficients[i] * math.Pow(x, float64(i))
	}
	return uint64(y)
}

// GuessSDXLRefinerDiffusionModelMemoryUsage returns the memory usage in bytes for the given width and height,
// which is calculated by linear regression or polynomial regression.
func GuessSDXLRefinerDiffusionModelMemoryUsage(width, height uint32, flashAttention bool) uint64 {
	coefficients := []float64{4.939599234485548e+07, 155.2477810191175, 0.0007351735797614931}
	degree := 2
	x := float64(width * height)

	if flashAttention {
		coefficients = []float64{7.0313433199802125e+06, 599.4137437226634}
		degree = 1
	}

	y := float64(0)
	for i := 0; i <= degree; i++ {
		y += coefficients[i] * math.Pow(x, float64(i))
	}
	return uint64(y)
}

// GuessSD3MediumDiffusionModelMemoryUsage returns the memory usage in bytes for the given width and height,
// which is calculated by linear regression or polynomial regression.
func GuessSD3MediumDiffusionModelMemoryUsage(width, height uint32, flashAttention bool) uint64 {
	coefficients := []float64{1.6529921370035086e+07, 234.66562477184195, 0.0014648995324747492}
	degree := 2
	x := float64(width * height)

	y := float64(0)
	for i := 0; i <= degree; i++ {
		y += coefficients[i] * math.Pow(x, float64(i))
	}
	return uint64(y)
}

// GuessSD35MediumDiffusionModelMemoryUsage returns the memory usage in bytes for the given width and height,
// which is calculated by linear regression or polynomial regression.
func GuessSD35MediumDiffusionModelMemoryUsage(width, height uint32, flashAttention bool) uint64 {
	coefficients := []float64{1.7441103472644456e+07, 281.695681980568, 0.0014651233076620938}
	degree := 2
	x := float64(width * height)

	y := float64(0)
	for i := 0; i <= degree; i++ {
		y += coefficients[i] * math.Pow(x, float64(i))
	}
	return uint64(y)
}

// GuessSD35LargeDiffusionModelMemoryUsage returns the memory usage in bytes for the given width and height,
// which is calculated by linear regression or polynomial regression.
func GuessSD35LargeDiffusionModelMemoryUsage(width, height uint32, flashAttention bool) uint64 {
	coefficients := []float64{2.320436920291992e+07, 410.3731196298318, 0.002319594715894278}
	degree := 2
	x := float64(width * height)

	y := float64(0)
	for i := 0; i <= degree; i++ {
		y += coefficients[i] * math.Pow(x, float64(i))
	}
	return uint64(y)
}

// GuessFLUXDiffusionModelMemoryUsage returns the memory usage in bytes for the given width and height,
// which is calculated by linear regression or polynomial regression.
func GuessFLUXDiffusionModelMemoryUsage(width, height uint32, flashAttention bool) uint64 {
	coefficients := []float64{4.651166867423782e+07, 997.7758807792155, 0.001457339256095295}
	degree := 2
	x := float64(width * height)

	y := float64(0)
	for i := 0; i <= degree; i++ {
		y += coefficients[i] * math.Pow(x, float64(i))
	}
	return uint64(y)
}
