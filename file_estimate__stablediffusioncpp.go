package gguf_parser

import (
	"regexp"

	"github.com/gpustack/gguf-parser-go/util/ptr"
	"github.com/gpustack/gguf-parser-go/util/stringx"
)

// Types for StableDiffusionCpp estimation.
type (
	// StableDiffusionCppRunEstimate represents the estimated result of loading the GGUF file in stable-diffusion.cpp.
	StableDiffusionCppRunEstimate struct {
		// Type describes what type this GGUF file is.
		Type string `json:"type"`
		// Architecture describes what architecture this GGUF file implements.
		//
		// All lowercase ASCII.
		Architecture string `json:"architecture"`
		// FlashAttention is the flag to indicate whether enable the flash attention,
		// true for enable.
		FlashAttention bool `json:"flashAttention"`
		// FullOffloaded is the flag to indicate whether the layers are fully offloaded,
		// false for partial offloaded or zero offloaded.
		FullOffloaded bool `json:"fullOffloaded"`
		// NoMMap is the flag to indicate whether support the mmap,
		// true for support.
		NoMMap bool `json:"noMMap"`
		// Distributable is the flag to indicate whether the model is distributable,
		// true for distributable.
		Distributable bool `json:"distributable"`
		// Devices represents the usage for running the GGUF file,
		// the first device is the CPU, and the rest are GPUs.
		Devices []StableDiffusionCppRunDeviceUsage `json:"devices"`
		// Autoencoder is the estimated result of the autoencoder.
		Autoencoder *StableDiffusionCppRunEstimate `json:"autoencoder,omitempty"`
		// Conditioners is the estimated result of the conditioners.
		Conditioners []StableDiffusionCppRunEstimate `json:"conditioners,omitempty"`
		// Upscaler is the estimated result of the upscaler.
		Upscaler *StableDiffusionCppRunEstimate `json:"upscaler,omitempty"`
		// ControlNet is the estimated result of the control net.
		ControlNet *StableDiffusionCppRunEstimate `json:"controlNet,omitempty"`
	}

	// StableDiffusionCppRunDeviceUsage represents the usage for running the GGUF file in llama.cpp.
	StableDiffusionCppRunDeviceUsage struct {
		// Remote is the flag to indicate whether the device is remote,
		// true for remote.
		Remote bool `json:"remote"`
		// Position is the relative position of the device,
		// starts from 0.
		//
		// If Remote is true, Position is the position of the remote devices,
		// Otherwise, Position is the position of the device in the local devices.
		Position int `json:"position"`
		// Footprint is the memory footprint for bootstrapping.
		Footprint GGUFBytesScalar `json:"footprint"`
		// Parameter is the running parameters that the device processes.
		Parameter GGUFParametersScalar `json:"parameter"`
		// Weight is the memory usage of weights that the device loads.
		Weight GGUFBytesScalar `json:"weight"`
		// Computation is the memory usage of computation that the device processes.
		Computation GGUFBytesScalar `json:"computation"`
	}
)

func (gf *GGUFFile) EstimateStableDiffusionCppRun(opts ...GGUFRunEstimateOption) (e StableDiffusionCppRunEstimate) {
	// Options
	var o _GGUFRunEstimateOptions
	for _, opt := range opts {
		opt(&o)
	}
	switch {
	case o.TensorSplitFraction == nil:
		o.TensorSplitFraction = []float64{1}
		o.MainGPUIndex = 0
	case o.MainGPUIndex < 0 || o.MainGPUIndex >= len(o.TensorSplitFraction):
		panic("main device index must be range of 0 to the length of tensor split fraction")
	}
	if len(o.DeviceMetrics) > 0 {
		for i, j := 0, len(o.DeviceMetrics)-1; i < len(o.TensorSplitFraction)-j; i++ {
			o.DeviceMetrics = append(o.DeviceMetrics, o.DeviceMetrics[j])
		}
		o.DeviceMetrics = o.DeviceMetrics[:len(o.TensorSplitFraction)+1]
	}
	if o.SDCBatchCount == nil {
		o.SDCBatchCount = ptr.To[int32](1)
	}
	if o.SDCHeight == nil {
		o.SDCHeight = ptr.To[uint32](512)
	}
	if o.SDCWidth == nil {
		o.SDCWidth = ptr.To[uint32](512)
	}
	if o.SDCOffloadConditioner == nil {
		o.SDCOffloadConditioner = ptr.To(true)
	}
	if o.SDCOffloadAutoencoder == nil {
		o.SDCOffloadAutoencoder = ptr.To(true)
	}
	if o.SDCAutoencoderTiling == nil {
		o.SDCAutoencoderTiling = ptr.To(true)
	}

	// Devices.
	e.Devices = make([]StableDiffusionCppRunDeviceUsage, len(o.TensorSplitFraction)+1)

	// Metadata.
	a := gf.Architecture()
	e.Type = a.Type
	e.Architecture = normalizeArchitecture(a.DiffusionArchitecture)

	// Flash attention.
	e.FlashAttention = false // TODO: Implement this.

	// Distributable.
	e.Distributable = false // TODO: Implement this.

	// Offload.
	e.FullOffloaded = true // TODO: Implement this.

	// NoMMap.
	e.NoMMap = true // TODO: Implement this.

	// Autoencoder.
	if a.DiffusionAutoencoder != nil {
		e.Autoencoder = &StableDiffusionCppRunEstimate{
			Type:           "model",
			Architecture:   e.Architecture + "_vae",
			FlashAttention: e.FlashAttention,
			Distributable:  e.Distributable,
			FullOffloaded:  e.FullOffloaded,
			NoMMap:         e.NoMMap,
			Devices:        make([]StableDiffusionCppRunDeviceUsage, len(e.Devices)),
		}
	}

	// Conditioners.
	if len(a.DiffusionConditioners) != 0 {
		e.Conditioners = make([]StableDiffusionCppRunEstimate, 0, len(a.DiffusionConditioners))
		for i := range a.DiffusionConditioners {
			e.Conditioners = append(e.Conditioners, StableDiffusionCppRunEstimate{
				Type:           "model",
				Architecture:   normalizeArchitecture(a.DiffusionConditioners[i].Architecture),
				FlashAttention: e.FlashAttention,
				Distributable:  e.Distributable,
				FullOffloaded:  e.FullOffloaded,
				NoMMap:         e.NoMMap,
				Devices:        make([]StableDiffusionCppRunDeviceUsage, len(e.Devices)),
			})
		}
	}

	// Footprint
	{
		// Bootstrap.
		e.Devices[0].Footprint = GGUFBytesScalar(5*1024*1024) /* model load */ + (gf.Size - gf.ModelSize) /* metadata */

		// Output buffer,
		// see
		// TODO: Implement this.
	}

	var cdLs, aeLs, cpLs GGUFLayerTensorInfos
	{
		var tis GGUFTensorInfos
		tis = gf.TensorInfos.Search(regexp.MustCompile(`^cond_stage_model\..*`))
		if len(tis) != 0 {
			cdLs = tis.Layers()
			if len(cdLs) != len(e.Conditioners) {
				panic("conditioners' layers count mismatch")
			}
			// NB(thxCode): resort the layers to match the order of the conditioners.
			cdLsSorted := make([]IGGUFTensorInfos, len(cdLs))
			cdLsSorted[0] = cdLs[len(cdLs)-1]
			for i := 1; i < len(cdLs); i++ {
				cdLsSorted[i] = cdLs[i-1]
			}
			cdLs = cdLsSorted
		}
		tis = gf.TensorInfos.Search(regexp.MustCompile(`^first_stage_model\..*`))
		if len(tis) != 0 {
			aeLs = tis.Layers()
		}
		tis = gf.TensorInfos.Search(regexp.MustCompile(`^model\.diffusion_model\..*`))
		if len(tis) != 0 {
			cpLs = tis.Layers()
		} else {
			cpLs = gf.TensorInfos.Layers()
		}
	}

	// Weight & Parameter.
	{
		// Conditioners.
		if cdLs != nil {
			d := 0
			if *o.SDCOffloadConditioner {
				d = 1
			}
			for i := range cdLs {
				e.Conditioners[i].Devices[d].Weight = GGUFBytesScalar(cdLs[i].Bytes())
				e.Conditioners[i].Devices[d].Parameter = GGUFParametersScalar(cdLs[i].Elements())
			}
		}

		// Autoencoder.
		if aeLs != nil {
			d := 0
			if *o.SDCOffloadAutoencoder {
				d = 1
			}
			e.Autoencoder.Devices[d].Weight = GGUFBytesScalar(aeLs.Bytes())
			e.Autoencoder.Devices[d].Parameter = GGUFParametersScalar(aeLs.Elements())
		}

		// Compute.
		if cpLs != nil {
			e.Devices[1].Weight = GGUFBytesScalar(cpLs.Bytes())
			e.Devices[1].Parameter = GGUFParametersScalar(cpLs.Elements())
		}
	}

	// Computation.
	{
		// TODO: Implement this.
	}

	return e
}

// Types for StableDiffusionCpp estimated summary.
type (
	// StableDiffusionCppRunEstimateSummary represents the estimated summary of loading the GGUF file in stable-diffusion.cpp.
	StableDiffusionCppRunEstimateSummary struct {
		/* Basic */

		// Items
		Items []StableDiffusionCppRunEstimateSummaryItem `json:"items"`

		/* Appendix */

		// Type describes what type this GGUF file is.
		Type string `json:"type"`
		// Architecture describes what architecture this GGUF file implements.
		//
		// All lowercase ASCII.
		Architecture string `json:"architecture"`
		// FlashAttention is the flag to indicate whether enable the flash attention,
		// true for enable.
		FlashAttention bool `json:"flashAttention"`
		// NoMMap is the flag to indicate whether the file must be loaded without mmap,
		// true for total loaded.
		NoMMap bool `json:"noMMap"`
		// Distributable is the flag to indicate whether the model is distributable,
		// true for distributable.
		Distributable bool `json:"distributable"`
	}

	// StableDiffusionCppRunEstimateSummaryItem represents the estimated summary item of loading the GGUF file in stable-diffusion.cpp.
	StableDiffusionCppRunEstimateSummaryItem struct {
		// FullOffloaded is the flag to indicate whether the layers are fully offloaded,
		// false for partial offloaded or zero offloaded.
		FullOffloaded bool `json:"fullOffloaded"`
		// RAM is the memory usage for loading the GGUF file in RAM.
		RAM StableDiffusionCppRunEstimateMemory `json:"ram"`
		// VRAMs is the memory usage for loading the GGUF file in VRAM per device.
		VRAMs []StableDiffusionCppRunEstimateMemory `json:"vrams"`
	}

	// StableDiffusionCppRunEstimateMemory represents the memory usage for loading the GGUF file in llama.cpp.
	StableDiffusionCppRunEstimateMemory struct {
		// Remote is the flag to indicate whether the device is remote,
		// true for remote.
		Remote bool `json:"remote"`
		// Position is the relative position of the device,
		// starts from 0.
		//
		// If Remote is true, Position is the position of the remote devices,
		// Otherwise, Position is the position of the device in the local devices.
		Position int `json:"position"`
		// UMA represents the usage of Unified Memory Architecture.
		UMA GGUFBytesScalar `json:"uma"`
		// NonUMA represents the usage of Non-Unified Memory Architecture.
		NonUMA GGUFBytesScalar `json:"nonuma"`
	}
)

// SummarizeItem returns the corresponding LLaMACppRunEstimateSummaryItem with the given options.
func (e StableDiffusionCppRunEstimate) SummarizeItem(
	mmap bool,
	nonUMARamFootprint, nonUMAVramFootprint uint64,
) (emi StableDiffusionCppRunEstimateSummaryItem) {
	// RAM.
	{
		fp := e.Devices[0].Footprint
		wg := e.Devices[0].Weight
		cp := e.Devices[0].Computation

		// UMA.
		emi.RAM.UMA = fp + wg + cp

		// NonUMA.
		emi.RAM.NonUMA = GGUFBytesScalar(nonUMARamFootprint) + emi.RAM.UMA
	}

	// VRAMs.
	emi.VRAMs = make([]StableDiffusionCppRunEstimateMemory, len(e.Devices)-1)
	{
		for i, d := range e.Devices[1:] {
			fp := d.Footprint
			wg := d.Weight
			cp := d.Computation

			// UMA.
			emi.VRAMs[i].UMA = fp + wg + cp

			// NonUMA.
			emi.VRAMs[i].NonUMA = GGUFBytesScalar(nonUMAVramFootprint) + emi.VRAMs[i].UMA
		}
	}

	// Add antoencoder's usage.
	if e.Autoencoder != nil {
		aemi := e.Autoencoder.SummarizeItem(mmap, 0, 0)
		emi.RAM.UMA += aemi.RAM.UMA
		emi.RAM.NonUMA += aemi.RAM.NonUMA
		for i, v := range aemi.VRAMs {
			emi.VRAMs[i].UMA += v.UMA
			emi.VRAMs[i].NonUMA += v.NonUMA
		}
	}

	// Add conditioners' usage.
	for i := range e.Conditioners {
		cemi := e.Conditioners[i].SummarizeItem(mmap, 0, 0)
		emi.RAM.UMA += cemi.RAM.UMA
		emi.RAM.NonUMA += cemi.RAM.NonUMA
		for i, v := range cemi.VRAMs {
			emi.VRAMs[i].UMA += v.UMA
			emi.VRAMs[i].NonUMA += v.NonUMA
		}
	}

	// Add upscaler's usage.
	if e.Upscaler != nil {
		uemi := e.Upscaler.SummarizeItem(mmap, 0, 0)
		emi.RAM.UMA += uemi.RAM.UMA
		emi.RAM.NonUMA += uemi.RAM.NonUMA
		for i, v := range uemi.VRAMs {
			emi.VRAMs[i].UMA += v.UMA
			emi.VRAMs[i].NonUMA += v.NonUMA
		}
	}

	// Add control net's usage.
	if e.ControlNet != nil {
		cnemi := e.ControlNet.SummarizeItem(mmap, 0, 0)
		emi.RAM.UMA += cnemi.RAM.UMA
		emi.RAM.NonUMA += cnemi.RAM.NonUMA
		for i, v := range cnemi.VRAMs {
			emi.VRAMs[i].UMA += v.UMA
			emi.VRAMs[i].NonUMA += v.NonUMA
		}
	}

	return emi
}

// Summarize returns the corresponding StableDiffusionCppRunEstimate with the given options.
func (e StableDiffusionCppRunEstimate) Summarize(
	mmap bool,
	nonUMARamFootprint, nonUMAVramFootprint uint64,
) (es StableDiffusionCppRunEstimateSummary) {
	// Items.
	es.Items = []StableDiffusionCppRunEstimateSummaryItem{
		e.SummarizeItem(mmap, nonUMARamFootprint, nonUMAVramFootprint),
	}

	// Just copy from the original estimate.
	es.Type = e.Type
	es.Architecture = e.Architecture
	es.FlashAttention = e.FlashAttention
	es.NoMMap = e.NoMMap
	es.Distributable = e.Distributable

	return es
}

func normalizeArchitecture(arch string) string {
	return stringx.ReplaceAllFunc(arch, func(r rune) rune {
		switch r {
		case ' ', '.', '-', '/', ':':
			return '_' // Replace with underscore.
		}
		if r >= 'A' && r <= 'Z' {
			r += 'a' - 'A' // Lowercase.
		}
		return r
	})
}
