package gguf_parser

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/thxcode/gguf-parser-go/util/httpx"
	"github.com/thxcode/gguf-parser-go/util/json"
	"github.com/thxcode/gguf-parser-go/util/stringx"
)

// Inspired by https://github.com/ollama/ollama/blob/380e06e5bea06ae8ded37f47c37bd5d604194d3e/types/model/name.go,
// and https://github.com/ollama/ollama/blob/380e06e5bea06ae8ded37f47c37bd5d604194d3e/server/modelpath.go.

const (
	OllamaDefaultScheme    = "https"
	OllamaDefaultRegistry  = "ollama.com"
	OllamaDefaultNamespace = "library"
	OllamaDefaultTag       = "latest"
)

type (
	OllamaModel struct {
		Schema        string             `json:"schema"`
		Registry      string             `json:"registry"`
		Namespace     string             `json:"namespace"`
		Repository    string             `json:"repository"`
		Tag           string             `json:"tag"`
		SchemaVersion uint32             `json:"schemaVersion"`
		MediaType     string             `json:"mediaType"`
		Config        OllamaModelLayer   `json:"config"`
		Layers        []OllamaModelLayer `json:"layers"`
	}
	OllamaModelLayer struct {
		MediaType string `json:"mediaType"`
		Size      uint64 `json:"size"`
		Digest    string `json:"digest"`

		model *OllamaModel
	}
)

// ParseOllamaModel parses the given Ollama model string,
// and returns the OllamaModel, or nil if the model is invalid.
func ParseOllamaModel(model string) *OllamaModel {
	if model == "" {
		return nil
	}

	om := OllamaModel{
		Schema:    OllamaDefaultScheme,
		Registry:  OllamaDefaultRegistry,
		Namespace: OllamaDefaultNamespace,
		Tag:       OllamaDefaultTag,
	}

	m := model

	// Drop digest.
	m, _, _ = stringx.CutFromRight(m, "@")

	// Get tag.
	m, s, ok := stringx.CutFromRight(m, ":")
	if ok && s != "" {
		om.Tag = s
	}

	// Get repository.
	m, s, ok = stringx.CutFromRight(m, "/")
	if ok && s != "" {
		om.Repository = s
	} else if m != "" {
		om.Repository = m
		m = ""
	}

	// Get namespace.
	m, s, ok = stringx.CutFromRight(m, "/")
	if ok && s != "" {
		om.Namespace = s
	} else if m != "" {
		om.Namespace = m
		m = ""
	}

	// Get registry.
	m, s, ok = stringx.CutFromLeft(m, "://")
	if ok && s != "" {
		om.Schema = m
		om.Registry = s
	} else if m != "" {
		om.Registry = m
	}

	if om.Repository == "" {
		return nil
	}
	return &om
}

func (om *OllamaModel) String() string {
	var b strings.Builder
	if om.Registry != "" {
		b.WriteString(om.Registry)
		b.WriteByte('/')
	}
	if om.Namespace != "" {
		b.WriteString(om.Namespace)
		b.WriteByte('/')
	}
	b.WriteString(om.Repository)
	if om.Tag != "" {
		b.WriteByte(':')
		b.WriteString(om.Tag)
	}
	return b.String()
}

// GetLayer returns the OllamaModelLayer with the given media type,
// and true if found, and false otherwise.
func (om *OllamaModel) GetLayer(mediaType string) (OllamaModelLayer, bool) {
	for i := range om.Layers {
		if om.Layers[i].MediaType == mediaType {
			return om.Layers[i], true
		}
	}
	return OllamaModelLayer{}, false
}

// SearchLayers returns a list of OllamaModelLayer with the media type that matches the given regex.
func (om *OllamaModel) SearchLayers(mediaTypeRegex *regexp.Regexp) []OllamaModelLayer {
	var ls []OllamaModelLayer
	for i := range om.Layers {
		if mediaTypeRegex.MatchString(om.Layers[i].MediaType) {
			ls = append(ls, om.Layers[i])
		}
	}
	return ls
}

// URL returns the URL of the OllamaModel.
func (om *OllamaModel) URL() *url.URL {
	u := &url.URL{
		Scheme: om.Schema,
		Host:   om.Registry,
	}
	return u.JoinPath("v2", om.Namespace, om.Repository, "manifests", om.Tag)
}

// WebURL returns the Ollama web URL of the OllamaModel.
func (om *OllamaModel) WebURL() *url.URL {
	u := &url.URL{
		Scheme: om.Schema,
		Host:   om.Registry,
	}
	return u.JoinPath(om.Namespace, om.Repository+":"+om.Tag)
}

// Complete completes the OllamaModel with the given context and http client.
func (om *OllamaModel) Complete(ctx context.Context, cli *http.Client) error {
	req, err := httpx.NewGetRequestWithContext(ctx, om.URL().String())
	if err != nil {
		return fmt.Errorf("new request: %w", err)
	}

	err = httpx.Do(cli, req, func(resp *http.Response) error {
		if resp.StatusCode != http.StatusOK {
			return fmt.Errorf("status code %d", resp.StatusCode)
		}
		return json.NewDecoder(resp.Body).Decode(om)
	})
	if err != nil {
		return fmt.Errorf("do request: %w", err)
	}

	// Connect.
	om.Config.model = om
	for i := range om.Layers {
		om.Layers[i].model = om
	}

	return nil
}

// URL returns the URL of the OllamaModelLayer.
func (ol *OllamaModelLayer) URL() *url.URL {
	if ol.model == nil {
		return nil
	}

	u := &url.URL{
		Scheme: ol.model.Schema,
		Host:   ol.model.Registry,
	}
	return u.JoinPath("v2", ol.model.Namespace, ol.model.Repository, "blobs", ol.Digest)
}

// WebURL returns the Ollama web URL of the OllamaModelLayer.
func (ol *OllamaModelLayer) WebURL() *url.URL {
	if ol.model == nil || len(ol.MediaType) < 12 {
		return nil
	}

	dg := strings.TrimPrefix(ol.Digest, "sha256:")[:12]
	u := &url.URL{
		Scheme: ol.model.Schema,
		Host:   ol.model.Registry,
	}
	return u.JoinPath(ol.model.Namespace, ol.model.Repository+":"+ol.model.Tag, "blobs", dg)
}
