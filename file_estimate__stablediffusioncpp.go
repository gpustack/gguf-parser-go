package gguf_parser

import (
	"regexp"

	"github.com/gpustack/gguf-parser-go/util/ptr"
	"github.com/gpustack/gguf-parser-go/util/stringx"
	"strings"
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
		// ImageOnly is the flag to indicate whether the model is used for generating image,
		// true for embedding only.
		ImageOnly bool `json:"imageOnly"`
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
		o.SDCAutoencoderTiling = ptr.To(false)
	}

	// Devices.
	e.Devices = make([]StableDiffusionCppRunDeviceUsage, len(o.TensorSplitFraction)+1)

	// Metadata.
	a := gf.Architecture()
	e.Type = a.Type
	e.Architecture = normalizeArchitecture(a.DiffusionArchitecture)

	// Flash attention.
	if o.FlashAttention && !strings.HasPrefix(a.DiffusionArchitecture, "Stable Diffusion 3") {
		// NB(thxCode): Stable Diffusion 3 doesn't support flash attention yet,
		// see https://github.com/leejet/stable-diffusion.cpp/pull/386.
		e.FlashAttention = true
	}

	// Distributable.
	e.Distributable = false // TODO: Implement this.

	// Offload.
	e.FullOffloaded = true // TODO: Implement this.

	// NoMMap.
	e.NoMMap = true // TODO: Implement this.

	// ImageOnly.
	e.ImageOnly = true // TODO: Implement this.

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
		e.Devices[0].Footprint = GGUFBytesScalar(10*1024*1024) /* model load */ + (gf.Size - gf.ModelSize) /* metadata */

		// Output buffer,
		// see
		// TODO: Implement this.
	}

	var cdLs, aeLs, mdLs GGUFLayerTensorInfos
	{
		var tis GGUFTensorInfos
		tis = gf.TensorInfos.Search(regexp.MustCompile(`^cond_stage_model\..*`))
		if len(tis) != 0 {
			cdLs = tis.Layers()
			if len(cdLs) != len(e.Conditioners) {
				panic("conditioners' layers count mismatch")
			}
		}
		tis = gf.TensorInfos.Search(regexp.MustCompile(`^first_stage_model\..*`))
		if len(tis) != 0 {
			aeLs = tis.Layers()
		}
		tis = gf.TensorInfos.Search(regexp.MustCompile(`^model\.diffusion_model\..*`))
		if len(tis) != 0 {
			mdLs = tis.Layers()
		} else {
			mdLs = gf.TensorInfos.Layers()
		}
	}

	var cdDevIdx, aeDevIdx, mdDevIdx int
	{
		if *o.SDCOffloadConditioner {
			cdDevIdx = 1
		}
		if *o.SDCOffloadAutoencoder {
			aeDevIdx = 1
		}
		mdDevIdx = 1
	}

	// Weight & Parameter.
	{
		// Conditioners.
		for i := range cdLs {
			e.Conditioners[i].Devices[cdDevIdx].Weight = GGUFBytesScalar(cdLs[i].Bytes())
			e.Conditioners[i].Devices[cdDevIdx].Parameter = GGUFParametersScalar(cdLs[i].Elements())
		}

		// Autoencoder.
		if aeLs != nil {
			e.Autoencoder.Devices[aeDevIdx].Weight = GGUFBytesScalar(aeLs.Bytes())
			e.Autoencoder.Devices[aeDevIdx].Parameter = GGUFParametersScalar(aeLs.Elements())
		}

		// Model.
		if mdLs != nil {
			e.Devices[mdDevIdx].Weight = GGUFBytesScalar(mdLs.Bytes())
			e.Devices[mdDevIdx].Parameter = GGUFParametersScalar(mdLs.Elements())
		}
	}

	// Computation.
	{
		// Bootstrap, compute metadata,
		// see https://github.com/ggerganov/llama.cpp/blob/d6ef0e77dd25f54fb5856af47e3926cf6f36c281/llama.cpp#L16135-L16136.
		cm := GGMLTensorOverhead()*GGMLComputationGraphNodesMaximum +
			GGMLComputationGraphOverhead(GGMLComputationGraphNodesMaximum, false)
		e.Devices[0].Computation = GGUFBytesScalar(cm)

		// Work context,
		// see https://github.com/leejet/stable-diffusion.cpp/blob/4570715727f35e5a07a76796d823824c8f42206c/stable-diffusion.cpp#L1467-L1481,
		//     https://github.com/leejet/stable-diffusion.cpp/blob/4570715727f35e5a07a76796d823824c8f42206c/stable-diffusion.cpp#L1572-L1586,
		//     https://github.com/leejet/stable-diffusion.cpp/blob/4570715727f35e5a07a76796d823824c8f42206c/stable-diffusion.cpp#L1675-L1679,
		//     https://github.com/thxCode/stable-diffusion.cpp/blob/78629d6340f763a8fe14372e0ba3ace73526a265/stable-diffusion.cpp#L2185-L2189,
		//     https://github.com/thxCode/stable-diffusion.cpp/blob/78629d6340f763a8fe14372e0ba3ace73526a265/stable-diffusion.cpp#L2270-L2274.
		//
		{
			var wcSize uint32 = 50 * 1024 * 1024
			wcSize += *o.SDCWidth * *o.SDCHeight * 3 * 4 /* sizeof(float) */ * 2 // RGB
			e.Devices[0].Computation += GGUFBytesScalar(wcSize * uint32(ptr.Deref(o.ParallelSize, 1)))
		}

		// Conditioner learned conditions,
		// see https://github.com/leejet/stable-diffusion.cpp/blob/4570715727f35e5a07a76796d823824c8f42206c/conditioner.hpp#L388-L391,
		//     https://github.com/leejet/stable-diffusion.cpp/blob/4570715727f35e5a07a76796d823824c8f42206c/conditioner.hpp#L758-L766,
		//     https://github.com/leejet/stable-diffusion.cpp/blob/4570715727f35e5a07a76796d823824c8f42206c/conditioner.hpp#L1083-L1085.
		switch {
		case strings.HasPrefix(a.DiffusionArchitecture, "FLUX"):
			for i := range cdLs {
				ds := []uint64{1}
				switch i {
				case 0:
					ds = []uint64{768, 77}
				case 1:
					ds = []uint64{4096, 256}
				}
				cds := GGUFBytesScalar(GGMLTypeF32.RowSizeOf(ds)) * 2 // include unconditioner
				e.Conditioners[i].Devices[cdDevIdx].Computation += cds
			}
		case strings.HasPrefix(a.DiffusionArchitecture, "Stable Diffusion 3"):
			for i := range cdLs {
				ds := []uint64{1}
				switch i {
				case 0:
					ds = []uint64{768, 77}
				case 1:
					ds = []uint64{1280, 77}
				case 2:
					ds = []uint64{4096, 77}
				}
				cds := GGUFBytesScalar(GGMLTypeF32.RowSizeOf(ds)) * 2 // include unconditioner
				e.Conditioners[i].Devices[cdDevIdx].Computation += cds
			}
		default:
			for i := range cdLs {
				ds := []uint64{1}
				switch i {
				case 0:
					ds = []uint64{768, 77}
					if strings.HasSuffix(a.DiffusionArchitecture, "Refiner") {
						ds = []uint64{1280, 77}
					}
				case 1:
					ds = []uint64{1280, 77}
				}
				cds := GGUFBytesScalar(GGMLTypeF32.RowSizeOf(ds)) * 2 // include unconditioner
				e.Conditioners[i].Devices[cdDevIdx].Computation += cds
			}
		}

		// Diffusion nosier,
		// see https://github.com/leejet/stable-diffusion.cpp/blob/4570715727f35e5a07a76796d823824c8f42206c/stable-diffusion.cpp#L1361.
		{
			mds := GGUFBytesScalar(GGMLTypeF32.RowSizeOf([]uint64{uint64(*o.SDCWidth / 8), uint64(*o.SDCHeight / 8), 16, 1}))
			e.Devices[mdDevIdx].Computation += mds
		}

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
		// ImageOnly is the flag to indicate whether the model is used for generating image,
		// true for embedding only.
		ImageOnly bool `json:"imageOnly"`
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
	emi.FullOffloaded = e.FullOffloaded

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
	es.ImageOnly = e.ImageOnly
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
